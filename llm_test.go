package main

import (
	"fmt"
	"testing"
	"time"
)

// failNTimesClient returns an error for the first N calls, then succeeds.
type failNTimesClient struct {
	failures  int
	callCount int
	err       error
}

func (f *failNTimesClient) GetEmbedding(_ string) ([]float64, error) {
	f.callCount++
	if f.callCount <= f.failures {
		return nil, f.err
	}
	return []float64{0.1, 0.2, 0.3}, nil
}

func (f *failNTimesClient) Chat(_ []Message) (string, error) {
	return "", nil
}

func noopSleep(_ time.Duration) {}

func TestGetEmbeddingWithRetry_Success(t *testing.T) {
	client := &failNTimesClient{failures: 0}
	embedding, err := doEmbeddingWithRetry(client, "test", 3, noopSleep)
	if err != nil {
		t.Fatalf("expected no error, got: %v", err)
	}
	if len(embedding) != 3 {
		t.Fatalf("expected 3-dim embedding, got %d", len(embedding))
	}
	if client.callCount != 1 {
		t.Fatalf("expected 1 call, got %d", client.callCount)
	}
}

func TestGetEmbeddingWithRetry_RetryThenSuccess(t *testing.T) {
	client := &failNTimesClient{
		failures: 1,
		err:      fmt.Errorf("500 Internal Server Error"),
	}
	embedding, err := doEmbeddingWithRetry(client, "test", 3, noopSleep)
	if err != nil {
		t.Fatalf("expected success after retry, got: %v", err)
	}
	if len(embedding) != 3 {
		t.Fatalf("expected 3-dim embedding, got %d", len(embedding))
	}
	if client.callCount != 2 {
		t.Fatalf("expected 2 calls, got %d", client.callCount)
	}
}

func TestGetEmbeddingWithRetry_NonRetryableError(t *testing.T) {
	client := &failNTimesClient{
		failures: 3,
		err:      fmt.Errorf("401 Unauthorized"),
	}
	_, err := doEmbeddingWithRetry(client, "test", 3, noopSleep)
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if client.callCount != 1 {
		t.Fatalf("expected 1 call (no retries for 401), got %d", client.callCount)
	}
}

func TestGetEmbeddingWithRetry_ExhaustsAttempts(t *testing.T) {
	client := &failNTimesClient{
		failures: 5,
		err:      fmt.Errorf("503 Service Unavailable"),
	}
	_, err := doEmbeddingWithRetry(client, "test", 3, noopSleep)
	if err == nil {
		t.Fatal("expected error after exhausting attempts, got nil")
	}
	if client.callCount != 3 {
		t.Fatalf("expected 3 calls, got %d", client.callCount)
	}
	expected := "failed after 3 attempts"
	if got := err.Error(); len(got) < len(expected) || got[:len(expected)] != expected {
		t.Fatalf("expected error starting with %q, got %q", expected, got)
	}
}

func TestIsRetryableError(t *testing.T) {
	tests := []struct {
		name     string
		err      string
		expected bool
	}{
		{"500 error", "500 Internal Server Error", true},
		{"502 error", "502 Bad Gateway", true},
		{"503 error", "503 Service Unavailable", true},
		{"504 error", "504 Gateway Timeout", true},
		{"429 rate limit", "429 Too Many Requests", true},
		{"rate limit text", "rate limit exceeded", true},
		{"401 unauthorized", "401 Unauthorized", false},
		{"403 forbidden", "403 Forbidden", false},
		{"400 bad request", "400 Bad Request", false},
		{"generic error", "connection refused", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := isRetryableError(fmt.Errorf("%s", tt.err))
			if got != tt.expected {
				t.Errorf("isRetryableError(%q) = %v, want %v", tt.err, got, tt.expected)
			}
		})
	}
}
