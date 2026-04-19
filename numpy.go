package gguf

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
)

// NpyArray holds data loaded from a .npy file.
type NpyArray struct {
	Data  []float32
	Shape []int
	Dtype string
}

// ReadNpy reads a numpy .npy file and returns the data as float32.
// Supports float32, float64, int32, int64 source dtypes (converts to float32).
func ReadNpy(path string) (*NpyArray, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	// Read magic: \x93NUMPY
	magic := make([]byte, 6)
	if _, err := io.ReadFull(f, magic); err != nil {
		return nil, fmt.Errorf("read magic: %w", err)
	}
	if magic[0] != 0x93 || string(magic[1:6]) != "NUMPY" {
		return nil, fmt.Errorf("not a numpy file")
	}

	// Version
	var major, minor uint8
	binary.Read(f, binary.LittleEndian, &major)
	binary.Read(f, binary.LittleEndian, &minor)

	// Header length
	var headerLen int
	if major == 1 {
		var hl uint16
		binary.Read(f, binary.LittleEndian, &hl)
		headerLen = int(hl)
	} else {
		var hl uint32
		binary.Read(f, binary.LittleEndian, &hl)
		headerLen = int(hl)
	}

	// Read header (Python dict string)
	headerBytes := make([]byte, headerLen)
	io.ReadFull(f, headerBytes)
	header := string(headerBytes)

	// Parse dtype
	dtype := "f4" // default float32
	if i := strings.Index(header, "'descr':"); i >= 0 {
		s := header[i+8:]
		s = strings.TrimLeft(s, " '\"")
		end := strings.IndexAny(s, "'\"")
		if end > 0 {
			dtype = s[:end]
		}
	}

	// Parse shape
	var shape []int
	if i := strings.Index(header, "'shape':"); i >= 0 {
		s := header[i+8:]
		start := strings.Index(s, "(")
		end := strings.Index(s, ")")
		if start >= 0 && end > start {
			shapeStr := s[start+1 : end]
			for _, dim := range strings.Split(shapeStr, ",") {
				dim = strings.TrimSpace(dim)
				if dim == "" {
					continue
				}
				n, _ := strconv.Atoi(dim)
				shape = append(shape, n)
			}
		}
	}

	// Calculate total elements
	nelem := 1
	for _, s := range shape {
		nelem *= s
	}

	// Read data based on dtype
	data := make([]float32, nelem)

	// Strip endian prefix
	dt := dtype
	if len(dt) > 0 && (dt[0] == '<' || dt[0] == '>' || dt[0] == '=') {
		dt = dt[1:]
	}

	switch dt {
	case "f4": // float32
		binary.Read(f, binary.LittleEndian, data)
	case "f8": // float64
		tmp := make([]float64, nelem)
		binary.Read(f, binary.LittleEndian, tmp)
		for i, v := range tmp {
			data[i] = float32(v)
		}
	case "i4": // int32
		tmp := make([]int32, nelem)
		binary.Read(f, binary.LittleEndian, tmp)
		for i, v := range tmp {
			data[i] = float32(v)
		}
	case "i8": // int64
		tmp := make([]int64, nelem)
		binary.Read(f, binary.LittleEndian, tmp)
		for i, v := range tmp {
			data[i] = float32(v)
		}
	default:
		return nil, fmt.Errorf("unsupported dtype: %s", dtype)
	}

	return &NpyArray{Data: data, Shape: shape, Dtype: dtype}, nil
}

// Rows returns the number of rows (first dimension).
func (a *NpyArray) Rows() int {
	if len(a.Shape) == 0 {
		return 0
	}
	return a.Shape[0]
}

// Cols returns the number of columns (second dimension, or 1 for 1D).
func (a *NpyArray) Cols() int {
	if len(a.Shape) < 2 {
		return 1
	}
	return a.Shape[1]
}

// Row returns a slice of the i-th row.
func (a *NpyArray) Row(i int) []float32 {
	cols := a.Cols()
	return a.Data[i*cols : (i+1)*cols]
}

// WriteNpy writes a float32 array to .npy format.
// Any tool that reads numpy (Python, Julia, R, MATLAB) can load it.
func WriteNpy(path string, data []float32, shape ...int) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	// Build shape string
	shapeStr := "("
	for i, s := range shape {
		if i > 0 {
			shapeStr += ", "
		}
		shapeStr += strconv.Itoa(s)
	}
	if len(shape) == 1 {
		shapeStr += ","
	}
	shapeStr += ")"

	// Header dict
	header := fmt.Sprintf("{'descr': '<f4', 'fortran_order': False, 'shape': %s, }", shapeStr)
	// Pad to 64-byte alignment (header + magic + version + headerLen = aligned)
	padLen := 64 - ((10 + len(header) + 1) % 64)
	if padLen == 64 {
		padLen = 0
	}
	header += strings.Repeat(" ", padLen) + "\n"

	// Magic
	f.Write([]byte{0x93, 'N', 'U', 'M', 'P', 'Y'})
	// Version 1.0
	f.Write([]byte{1, 0})
	// Header length
	binary.Write(f, binary.LittleEndian, uint16(len(header)))
	// Header
	f.Write([]byte(header))
	// Data
	binary.Write(f, binary.LittleEndian, data)

	return nil
}
