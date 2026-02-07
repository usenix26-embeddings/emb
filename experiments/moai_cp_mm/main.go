package main


import (
	"flag"
	"fmt"
	"math"
	"time"

	"github.com/schollz/progressbar/v3"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
	"github.com/tuneinsight/lattigo/v6/utils/sampling"
)

func main() {
	// 1. Create CKKS parameters
	params, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:            16,                                    // Ring degree = 2^14 = 16384
		LogQ:            []int{55, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40}, // Ciphertext modulus primes
		LogP:            []int{61, 61, 61},                     // Auxiliary modulus for key-switching
		LogDefaultScale: 40,                                    // Scaling factor = 2^45
	})
	if err != nil {
		panic(err)
	}

	fmt.Println("CKKS Parameters created")
	fmt.Printf("  Ring degree N: %d\n", params.N())
	fmt.Printf("  Max slots: %d\n", params.MaxSlots())
	fmt.Printf("  Max level: %d\n", params.MaxLevel())

	// 2. Key generation
	kgen := rlwe.NewKeyGenerator(params)
	sk := kgen.GenSecretKeyNew()
	pk := kgen.GenPublicKeyNew(sk)

	fmt.Println("Keys generated")

	// 3. Create encoder, encryptor, decryptor, evaluator
	encoder := ckks.NewEncoder(params)
	encryptor := rlwe.NewEncryptor(params, pk)
	decryptor := rlwe.NewDecryptor(params, sk)
	evaluator := ckks.NewEvaluator(params, nil) // No relinearization key needed for scalar mult

	// 4. Define dimensions (from command line args)
	seq_len := flag.Int("seq_len", 5, "Sequence length")
	d := flag.Int("d", 10, "Hidden dimension")
	V := flag.Int("V", 12, "Vocabulary size")
	numRuns := flag.Int("runs", 5, "Number of benchmark runs")
	flag.Parse()
	fmt.Println("Embedding Params")
	fmt.Printf("  Sequence Length (seq_len): %d\n", *seq_len)
	fmt.Printf("  Hidden Dimension (d): %d\n", *d)
	fmt.Printf("  Vocab Size (V): %d\n", *V)



	// 5. Create input matrix of size seq_len x V (each row is a random one-hot vector)
	fmt.Println("Creating one-hot vectors..")
	input := make([][]float64, *seq_len)
	for i := 0; i < *seq_len; i++ {
		input[i] = make([]float64, *V)
		hotIdx := int(sampling.RandUint64() % uint64(*V))
		input[i][hotIdx] = 1.0
	}

	// 6. Create a matrix of size d x V with small random floats
	fmt.Println("Creating embedding matrix..")
	matrix := make([][]float64, *d)
	for i := 0; i < *d; i++ {
		matrix[i] = make([]float64, *V)
		for j := 0; j < *V; j++ {
			matrix[i][j] = sampling.RandFloat64(-1.0, 1.0)
		}
	}

	smallMatrices := *V < 20 && *d < 20 && *seq_len < 20

	fmt.Printf("\nMatrix created: %d x %d (random values in [-1, 1])\n", *d, *V)
	if smallMatrices {
		for i := 0; i < *d; i++ {
			fmt.Printf("  [")
			for j := 0; j < *V; j++ {
				if j > 0 {
					fmt.Printf(" ")
				}
				fmt.Printf("%7.4f", matrix[i][j])
			}
			fmt.Printf("]\n")
		}
	}

	fmt.Printf("\nInput matrix (transposed): %d x %d\n", *V, *seq_len)
	if smallMatrices {
		for j := 0; j < *V; j++ {
			fmt.Printf("  [")
			for i := 0; i < *seq_len; i++ {
				if i > 0 {
					fmt.Printf(" ")
				}
				fmt.Printf("%1.0f", input[i][j])
			}
			fmt.Printf("]\n")
		}
	}

	// 7. Encode each column of input matrix into a separate plaintext
	plaintexts := make([]*rlwe.Plaintext, *V)
	for j := 0; j < *V; j++ {
		col := make([]float64, *seq_len)
		for i := 0; i < *seq_len; i++ {
			col[i] = input[i][j]
		}
		plaintexts[j] = ckks.NewPlaintext(params, 1) // encode at specific level
		if err := encoder.Encode(col, plaintexts[j]); err != nil {
			panic(err)
		}
	}

	fmt.Printf("\nEncoded %d plaintexts (one per vocabulary index)\n", len(plaintexts))

	// 8. Encrypt each plaintext into a separate ciphertext
	ciphertexts := make([]*rlwe.Ciphertext, *V)
	for i, pt := range plaintexts {
		ciphertexts[i], err = encryptor.EncryptNew(pt)
		if err != nil {
			panic(err)
		}
	}

	fmt.Printf("Encrypted %d ciphertexts\n", len(ciphertexts))
	fmt.Printf("  Each ciphertext level: %d\n", ciphertexts[0].Level())

	// Pre-allocate output ciphertexts once (avoid allocations in hot path)
	fmt.Println("Pre-allocated output ciphertext")
	ciphertexts_out := make([]*rlwe.Ciphertext, *d)
	for j := 0; j < *d; j++ {
		ciphertexts_out[j] = ckks.NewCiphertext(params, 1, ciphertexts[0].Level())
	}

	// Run benchmark multiple times
	durations := make([]float64, *numRuns)
	bar := progressbar.Default(int64(*numRuns), "Benchmarking")

	for run := 0; run < *numRuns; run++ {
		start := time.Now()
		for j := 0; j < *d; j++ {
			// Initialize with first term using in-place Mul
			if err := evaluator.Mul(ciphertexts[0], matrix[j][0], ciphertexts_out[j]); err != nil {
				panic(err)
			}
			// Accumulate remaining terms using MulThenAdd (no allocations)
			for i := 1; i < *V; i++ {
				if err := evaluator.MulThenAdd(ciphertexts[i], matrix[j][i], ciphertexts_out[j]); err != nil {
					panic(err)
				}
			}
		}
		durations[run] = time.Since(start).Seconds() * 1000 // convert to ms
		bar.Add(1)
	}

	// Compute mean
	var sum float64
	for _, dur := range durations {
		sum += dur
	}
	mean := sum / float64(*numRuns)

	fmt.Printf("\nComputed matrix product: %d output ciphertexts (each with %d elements)\n", *d, *seq_len)
	if *numRuns == 1 {
		fmt.Printf("  Latency: %.2f ms\n", mean)
	} else {
		// Compute standard deviation
		var sqDiffSum float64
		for _, dur := range durations {
			sqDiffSum += (dur - mean) * (dur - mean)
		}
		std := math.Sqrt(sqDiffSum / float64(*numRuns))
		fmt.Printf("  Latency over %d runs: %.2f ± %.2f ms\n", *numRuns, mean, std)
	}

	// Decrypt and print each output ciphertext
	fmt.Printf("\nDecrypted output matrix: %d x %d\n", *d, *seq_len)
	if smallMatrices {
		for j := 0; j < *d; j++ {
			pt := decryptor.DecryptNew(ciphertexts_out[j])
			result := make([]float64, params.MaxSlots())
			if err := encoder.Decode(pt, result); err != nil {
				panic(err)
			}
			fmt.Printf("  [")
			for i := 0; i < *seq_len; i++ {
				if i > 0 {
					fmt.Printf(" ")
				}
				fmt.Printf("%7.4f", result[i])
			}
			fmt.Printf("]\n")
		}
	}
}
