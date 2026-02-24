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

// testNetError implements net.Error for testing network error retries.
type testNetError struct{ msg string }

func (e *testNetError) Error() string   { return e.msg }
func (e *testNetError) Timeout() bool   { return false }
func (e *testNetError) Temporary() bool { return true }

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
		err:      &APIError{StatusCode: 500, Status: "500 Internal Server Error"},
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
		err:      &APIError{StatusCode: 401, Status: "401 Unauthorized"},
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
		err:      &APIError{StatusCode: 503, Status: "503 Service Unavailable"},
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

func TestGetEmbeddingWithRetry_NetworkError(t *testing.T) {
	client := &failNTimesClient{
		failures: 1,
		err:      &testNetError{msg: "connection refused"},
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

func TestIsRetryableError(t *testing.T) {
	tests := []struct {
		name     string
		err      error
		expected bool
	}{
		{"500 error", &APIError{StatusCode: 500}, true},
		{"502 error", &APIError{StatusCode: 502}, true},
		{"503 error", &APIError{StatusCode: 503}, true},
		{"504 error", &APIError{StatusCode: 504}, true},
		{"429 rate limit", &APIError{StatusCode: 429}, true},
		{"401 unauthorized", &APIError{StatusCode: 401}, false},
		{"403 forbidden", &APIError{StatusCode: 403}, false},
		{"400 bad request", &APIError{StatusCode: 400}, false},
		{"network error", &testNetError{msg: "connection refused"}, true},
		{"generic error", fmt.Errorf("some error"), false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := isRetryableError(tt.err)
			if got != tt.expected {
				t.Errorf("isRetryableError(%v) = %v, want %v", tt.err, got, tt.expected)
			}
		})
	}
}
