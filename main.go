// Copyright 2017 The KCStat Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"fmt"
	"io/ioutil"

	"github.com/pointlander/compress"
	ga "github.com/pointlander/go-galib"
)

const (
	// Symbols the number of symbols
	Symbols = 256
	// Width the width of the genome
	Width = Symbols * Symbols
	// Verify verify the fitness function
	Verify = false
)

var data []byte

func press(g *ga.GAFloat32Genome) float32 {
	newCDF := func(size int) *compress.CDF16 {
		if size != 256 {
			panic("size is not 256")
		}
		cdf, mixin := make([]uint16, size+1), make([][]uint16, size)

		sum := 0
		for i := range cdf {
			cdf[i] = uint16(sum)
			sum += 32
		}

		for i := range mixin {
			sum, m, offset, total := 0, make([]uint16, size+1), i*Symbols, float32(0)
			for j := 0; j < Symbols; j++ {
				total += g.Gene[offset+j]
			}
			for j := range m[:Symbols] {
				m[j] = uint16(sum)
				sum += int(1 + (g.Gene[offset+j] * (compress.CDF16Scale - Symbols) / total))
				if Verify && j > 0 {
					if a, b := m[j], m[j-1]; a < b {
						panic(fmt.Sprintf("invalid mixin cdf %v,%v < %v,%v; sum=%v", j, a, j-1, b, sum))
					} else if a == b {
						panic(fmt.Sprintf("invalid mixin cdf %v,%v = %v,%v; sum=%v", j, a, j-1, b, sum))
					}
				}
			}
			m[Symbols] = compress.CDF16Scale
			mixin[i] = m
		}

		return &compress.CDF16{
			CDF:    cdf,
			Mixin:  mixin,
			Verify: Verify,
		}
	}

	input := make([]uint16, len(data))
	for i := range data {
		input[i] = uint16(data[i])
	}
	symbols, buffer := make(chan []uint16, 1), &bytes.Buffer{}
	symbols <- input
	close(symbols)
	bits := compress.Coder16{Alphabit: 256, Input: symbols}.FilteredAdaptiveCoder(newCDF).Code(buffer)

	if Verify {
		out, i := make([]byte, len(data)), 0
		output := func(symbol uint16) bool {
			out[i] = byte(symbol)
			i++
			return i >= len(out)
		}
		compress.Coder16{Alphabit: 256, Output: output}.FilteredAdaptiveDecoder(newCDF).Decode(buffer)

		if bytes.Compare(out, data) != 0 {
			panic("decompression failed")
		}
	}
	return float32(bits)
}

func main() {
	alice, err := ioutil.ReadFile("alice30.txt")
	if err != nil {
		panic(err)
	}
	data = alice

	stats, first := [Symbols][Symbols]int{}, ga.NewFloat32Genome(make([]float32, Width), press, 1, 0)
	for i := range alice[:len(alice)-Symbols] {
		for j := 0; j < Symbols; j++ {
			stats[alice[i]][alice[i+j]]++
		}
	}
	for i := range first.Gene {
		first.Gene[i] = .1
	}
	for i := 0; i < Symbols; i++ {
		total, offset := 0, i*256
		for j := 0; j < Symbols; j++ {
			total += stats[i][j]
		}
		if total == 0 {
			continue
		}
		for j := 0; j < Symbols; j++ {
			first.Gene[offset+j] = float32(stats[i][j]) / float32(total)
		}
	}

	input := make([]uint16, len(data))
	for i := range data {
		input[i] = uint16(data[i])
	}
	symbols, buffer := make(chan []uint16, 1), &bytes.Buffer{}
	symbols <- input
	close(symbols)
	bits := compress.Coder16{Alphabit: 256, Input: symbols}.FilteredAdaptiveCoder(compress.NewCDF16).Code(buffer)
	fmt.Printf("size = %.3f %.3f\n", float64(bits)/8, float64(bits)/(8*float64(len(data))))

	mutator := ga.NewMultiMutator()
	msh := new(ga.GAShiftMutator)
	msw := new(ga.GASwitchMutator)
	gm := NewGABoundedGaussianMutator(0.1, 0)
	mutator.Add(msh)
	mutator.Add(msw)
	mutator.Add(gm)

	param := ga.GAParameter{
		Initializer: new(GACDF2Initializer),
		Selector:    ga.NewGATournamentSelector(0.2, 5),
		PMutate:     0.5,
		PBreed:      0.2,
		Breeder:     new(ga.GA2PointBreeder),
		Mutator:     mutator,
	}
	gao := ga.NewGA(param)
	gao.Parallel = true
	gao.Init(200, first)

	gao.OptimizeUntil(func(best ga.GAGenome) bool {
		score := best.Score() / 8
		fmt.Printf("best = %.3f %.3f %.3f\n",
			score,
			score/float64(len(data)),
			8*score/float64(bits))
		return score < .3*float64(len(alice))
	})
}
