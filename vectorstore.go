package main

import (
	"compress/gzip"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"sort"
	"strings"
)

// VectorStore is a simple in-memory vector database
type VectorStore struct {
	Chunks     []Chunk
	Embeddings [][]float64
	Metadata   VectorStoreMetadata
}

// VectorStoreMetadata tracks information about the indexed source
type VectorStoreMetadata struct {
	IndexedAt    string        `json:"indexed_at"`
	SourcePath   string        `json:"source_path"`
	FileCount    int           `json:"file_count"`
	ChunkCount   int           `json:"chunk_count"`
	IndexedFiles []string      `json:"indexed_files"` // list of all indexed file paths
	SkippedFiles []SkippedFile `json:"skipped_files"` // files that were skipped with reasons
}

// SkippedFile represents a file that was skipped during indexing
type SkippedFile struct {
	Path   string `json:"path"`
	Reason string `json:"reason"` // e.g., "too large (150KB)", "test file", "binary file"
	Size   int64  `json:"size"`   // file size in bytes
}

// SearchResult represents a chunk with its similarity score
type SearchResult struct {
	Chunk      Chunk
	Similarity float64
}

// NewVectorStore creates a new vector store
func NewVectorStore() *VectorStore {
	return &VectorStore{
		Chunks:     make([]Chunk, 0),
		Embeddings: make([][]float64, 0),
	}
}

// Add adds a chunk and its embedding to the store
func (vs *VectorStore) Add(chunk Chunk, embedding []float64) {
	vs.Chunks = append(vs.Chunks, chunk)
	vs.Embeddings = append(vs.Embeddings, embedding)
}

// Search finds the most similar chunks to the query embedding
func (vs *VectorStore) Search(queryEmbedding []float64, topK int) []SearchResult {
	var results []SearchResult

	// calculate cosine similarity for each chunk
	for i, embedding := range vs.Embeddings {
		similarity := cosineSimilarity(queryEmbedding, embedding)
		results = append(results, SearchResult{
			Chunk:      vs.Chunks[i],
			Similarity: similarity,
		})
	}

	// sort by similarity (descending)
	sort.Slice(results, func(i, j int) bool {
		return results[i].Similarity > results[j].Similarity
	})

	// return top k results
	if topK > len(results) {
		topK = len(results)
	}

	return results[:topK]
}

// Save saves the vector store to disk (gzip compressed if .lrindex extension)
func (vs *VectorStore) Save(filepath string) error {
	data, err := json.Marshal(vs)
	if err != nil {
		return err
	}

	// if filepath ends with .lrindex, save as gzipped
	if strings.HasSuffix(filepath, ".lrindex") {
		f, err := os.Create(filepath)
		if err != nil {
			return err
		}
		defer f.Close()

		gw := gzip.NewWriter(f)

		if _, err := gw.Write(data); err != nil {
			gw.Close()
			return err
		}

		// must close gzip writer to flush all data before file closes
		if err := gw.Close(); err != nil {
			return err
		}

		// sync file to disk before returning
		if err := f.Sync(); err != nil {
			return fmt.Errorf("failed to sync file to disk: %w", err)
		}

		return nil
	}

	// otherwise save as plain json (backward compatibility)
	return os.WriteFile(filepath, data, 0644)
}

// Load loads the vector store from disk (auto-detects gzip compression)
func (vs *VectorStore) Load(filepath string) error {
	f, err := os.Open(filepath)
	if err != nil {
		return err
	}
	defer f.Close()

	var reader io.Reader = f

	// if filepath ends with .lrindex or file starts with gzip magic bytes, decompress
	if strings.HasSuffix(filepath, ".lrindex") {
		gr, err := gzip.NewReader(f)
		if err != nil {
			return err
		}
		defer gr.Close()
		reader = gr
	} else {
		// try to detect gzip by magic bytes for backward compat
		header := make([]byte, 2)
		if n, _ := f.Read(header); n == 2 && header[0] == 0x1f && header[1] == 0x8b {
			// gzip magic bytes detected
			f.Seek(0, 0) // reset
			gr, err := gzip.NewReader(f)
			if err != nil {
				return err
			}
			defer gr.Close()
			reader = gr
		} else {
			f.Seek(0, 0) // reset for regular json reading
		}
	}

	data, err := io.ReadAll(reader)
	if err != nil {
		return err
	}
	return json.Unmarshal(data, vs)
}

// cosineSimilarity calculates the cosine similarity between two vectors
func cosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) {
		return 0
	}

	var dotProduct, normA, normB float64

	for i := 0; i < len(a); i++ {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return dotProduct / (math.Sqrt(normA) * math.Sqrt(normB))
}
