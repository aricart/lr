package main

import (
	"strings"
)

// Chunk represents a text chunk with metadata
type Chunk struct {
	Text     string
	Source   string
	Metadata map[string]string
}

// ChunkDocument splits a document into smaller chunks
// uses different strategies based on document type
func ChunkDocument(doc Document, maxChunkSize int) []Chunk {
	var chunks []Chunk
	docType := doc.Metadata["type"]

	var sections []string

	// choose chunking strategy based on document type
	if docType == "markdown" {
		// split by markdown headers
		sections = splitByHeaders(doc.Content)
	} else if docType == "go" || docType == "javascript" || docType == "typescript" ||
		docType == "python" || docType == "java" || docType == "c" {
		// split code by functions/methods
		sections = splitByFunctions(doc.Content)
	} else {
		// fallback: split by paragraphs
		sections = splitByParagraphs(doc.Content, maxChunkSize)
	}

	for i, section := range sections {
		// skip very small chunks (likely noise)
		if len(strings.TrimSpace(section)) < 50 {
			continue
		}

		// estimate tokens (rough: 1 token ≈ 4 characters)
		estimatedTokens := len(section) / 4

		// if section is too large even for aggressive splitting, truncate it
		// openai embedding limit is 8192 tokens, we use 5000 to be very safe
		if estimatedTokens > 5000 {
			// aggressively split by lines
			subChunks := splitByLines(section, 16000) // ~4000 tokens per chunk
			for j, subChunk := range subChunks {
				chunk := Chunk{
					Text:   subChunk,
					Source: doc.Source,
					Metadata: map[string]string{
						"source":      doc.Source,
						"type":        docType,
						"chunk_index": string(rune(i)) + "." + string(rune(j)),
					},
				}
				chunks = append(chunks, chunk)
			}
		} else if len(section) <= maxChunkSize {
			// section is small enough, use as is
			chunk := Chunk{
				Text:   section,
				Source: doc.Source,
				Metadata: map[string]string{
					"source":      doc.Source,
					"type":        docType,
					"chunk_index": string(rune(i)),
				},
			}
			chunks = append(chunks, chunk)
		} else {
			// split large sections by paragraphs
			subChunks := splitByParagraphs(section, maxChunkSize)
			for j, subChunk := range subChunks {
				chunk := Chunk{
					Text:   subChunk,
					Source: doc.Source,
					Metadata: map[string]string{
						"source":      doc.Source,
						"type":        docType,
						"chunk_index": string(rune(i)) + "." + string(rune(j)),
					},
				}
				chunks = append(chunks, chunk)
			}
		}
	}

	return chunks
}

// splitByHeaders splits content by markdown headers
func splitByHeaders(content string) []string {
	var sections []string
	lines := strings.Split(content, "\n")
	var currentSection strings.Builder

	for _, line := range lines {
		// check if line is a header (starts with #)
		if strings.HasPrefix(strings.TrimSpace(line), "#") {
			// save current section if not empty
			if currentSection.Len() > 0 {
				sections = append(sections, strings.TrimSpace(currentSection.String()))
				currentSection.Reset()
			}
		}
		currentSection.WriteString(line)
		currentSection.WriteString("\n")
	}

	// add last section
	if currentSection.Len() > 0 {
		sections = append(sections, strings.TrimSpace(currentSection.String()))
	}

	return sections
}

// splitByParagraphs splits content by paragraphs, keeping size under maxSize
func splitByParagraphs(content string, maxSize int) []string {
	var chunks []string
	paragraphs := strings.Split(content, "\n\n")
	var currentChunk strings.Builder

	for _, para := range paragraphs {
		// if adding this paragraph exceeds max size, save current chunk
		if currentChunk.Len()+len(para)+2 > maxSize && currentChunk.Len() > 0 {
			chunks = append(chunks, strings.TrimSpace(currentChunk.String()))
			currentChunk.Reset()
		}

		currentChunk.WriteString(para)
		currentChunk.WriteString("\n\n")

		// if single paragraph is too large, split it anyway
		if currentChunk.Len() > maxSize {
			chunks = append(chunks, strings.TrimSpace(currentChunk.String()))
			currentChunk.Reset()
		}
	}

	// add last chunk
	if currentChunk.Len() > 0 {
		chunks = append(chunks, strings.TrimSpace(currentChunk.String()))
	}

	return chunks
}

// splitByLines splits content by lines when other methods fail
func splitByLines(content string, maxSize int) []string {
	var chunks []string
	lines := strings.Split(content, "\n")
	var currentChunk strings.Builder

	for _, line := range lines {
		// if adding this line exceeds max size, save current chunk
		if currentChunk.Len()+len(line)+1 > maxSize && currentChunk.Len() > 0 {
			chunks = append(chunks, currentChunk.String())
			currentChunk.Reset()
		}

		currentChunk.WriteString(line)
		currentChunk.WriteString("\n")
	}

	// add last chunk
	if currentChunk.Len() > 0 {
		chunks = append(chunks, currentChunk.String())
	}

	return chunks
}

// splitByFunctions attempts to split code by function/method definitions
func splitByFunctions(content string) []string {
	var sections []string
	lines := strings.Split(content, "\n")
	var currentSection strings.Builder
	var braceCount int
	inFunction := false

	for i, line := range lines {
		trimmed := strings.TrimSpace(line)

		// detect function start (simple heuristic)
		// go: func keyword
		// js/ts: function keyword, arrow functions, method definitions
		// python: def keyword
		// java: public/private/protected methods, class definitions
		// c: function definitions with return type
		isFunctionStart := strings.HasPrefix(trimmed, "func ") ||
			strings.HasPrefix(trimmed, "function ") ||
			strings.HasPrefix(trimmed, "def ") ||
			strings.HasPrefix(trimmed, "class ") ||
			strings.HasPrefix(trimmed, "public ") ||
			strings.HasPrefix(trimmed, "private ") ||
			strings.HasPrefix(trimmed, "protected ") ||
			(strings.Contains(trimmed, "=> {") || strings.Contains(trimmed, "=>")) ||
			(i > 0 && strings.Contains(trimmed, "(") && strings.Contains(trimmed, ")") && strings.Contains(trimmed, "{"))

		if isFunctionStart && !inFunction && braceCount == 0 {
			// save previous section if not empty
			if currentSection.Len() > 0 {
				sections = append(sections, strings.TrimSpace(currentSection.String()))
				currentSection.Reset()
			}
			inFunction = true
		}

		currentSection.WriteString(line)
		currentSection.WriteString("\n")

		// track braces to know when function ends
		braceCount += strings.Count(line, "{") - strings.Count(line, "}")

		// function ended
		if inFunction && braceCount == 0 && strings.Contains(line, "}") {
			sections = append(sections, strings.TrimSpace(currentSection.String()))
			currentSection.Reset()
			inFunction = false
		}
	}

	// add remaining content
	if currentSection.Len() > 0 {
		sections = append(sections, strings.TrimSpace(currentSection.String()))
	}

	// fallback: if we didn't find functions, split by blank lines
	if len(sections) <= 1 {
		return splitByParagraphs(content, 2000)
	}

	return sections
}
