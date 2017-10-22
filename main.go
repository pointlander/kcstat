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
	newCDF := func(size int) *compress.CDF {
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
				sum += int(1 + g.Gene[offset+j]*(compress.CDFScale-Symbols)/total)
			}
			m[Symbols] = compress.CDFScale
			mixin[i] = m
		}

		return &compress.CDF{
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

	mutator := ga.NewMultiMutator()
	msh := new(ga.GAShiftMutator)
	msw := new(ga.GASwitchMutator)
	gm := NewGABoundedGaussianMutator(0.3, 0)
	mutator.Add(msh)
	mutator.Add(msw)
	mutator.Add(gm)

	param := ga.GAParameter{
		Initializer: new(GACDFInitializer),
		Selector:    ga.NewGATournamentSelector(0.2, 5),
		PMutate:     0.5,
		PBreed:      0.2,
		Breeder:     new(ga.GA2PointBreeder),
		Mutator:     mutator,
	}
	gao := ga.NewGA(param)
	gao.Parallel = true
	gao.Init(200, ga.NewFloat32Genome(make([]float32, Width), press, 1, 0))

	gao.OptimizeUntil(func(best ga.GAGenome) bool {
		score := best.Score() / 8
		fmt.Printf("best = %v\n", score)
		return score < .3*float64(len(alice))
	})
}
