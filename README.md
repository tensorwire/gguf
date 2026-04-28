# gguf

Go-native model serialization. Read and write GGUF, SafeTensors, and NumPy formats. Zero dependencies beyond stdlib.

## Install

```bash
go get github.com/tensorwire/gguf
```

## Formats

| Format | Read | Write | Quantization |
|--------|------|-------|-------------|
| GGUF (llama.cpp/Ollama) | Yes | Yes | Q8_0, Q4_0, F16, F32 |
| SafeTensors (HuggingFace) | Yes | Yes | F16, BF16, F32, INT8, AWQ/GPTQ dequant |
| NumPy (.npy) | Yes | Yes | F32 |

## Usage

```go
// Read SafeTensors (HuggingFace models)
st, _ := gguf.OpenSafeTensors("/path/to/model")
data, _, _ := st.ReadTensorFloat32("model.layers.0.self_attn.q_proj.weight")

// Write GGUF with quantization
w := gguf.NewGGUFWriter()
w.AddMetadataString("general.architecture", "llama")
w.AddTensorQ8_0("blk.0.attn_q.weight", data, 4096, 4096)
w.Write("model-q8.gguf")

// Convert SafeTensors to GGUF
gguf.ConvertSafetensorsToGGUF("/path/to/model", "output.gguf", "q8_0")

// NumPy
arr, _ := gguf.ReadNpy("weights.npy")
gguf.WriteNpy("out.npy", data, 2, 3)
```

## Features

- Sharded SafeTensors support (multi-file models)
- Automatic F16/BF16 to F32 conversion on read
- AWQ and GPTQ dequantization
- Q8_0: 8-bit per-block quantization (~3.5x compression)
- Q4_0: 4-bit per-block quantization (~6.2x compression)
- Streaming reads for large models

## License

MIT
