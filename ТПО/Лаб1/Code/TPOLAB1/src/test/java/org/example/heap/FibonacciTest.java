package org.example.heap;

import org.junit.jupiter.api.Test;
import java.time.Duration;
import java.util.*;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assertions.assertTimeoutPreemptively;

public class FibonacciTest {

    @Test
    void testSequentialInsert5() {
        assertTimeoutPreemptively(Duration.ofSeconds(5), () -> {
            FibonacciHeap heap = new FibonacciHeap();
            for (int i = 10; i <= 15; i++) heap.insert(i);
            assertNotNull(heap.min());
            assertEquals(10, heap.min().key);
        });
    }

    @Test
    void testInsert100000() {
        assertTimeoutPreemptively(Duration.ofSeconds(5), () -> {
            FibonacciHeap heap = new FibonacciHeap();
            for (int i = 1; i <= 100000; i++) heap.insert(i);
            for (int i = 1; i <= 100000; i++) {
                assertEquals(i, heap.min().key);
                heap.removeMin();
            }
        });
    }

    @Test
    void testReversedInsert() {
        assertTimeoutPreemptively(Duration.ofSeconds(5), () -> {
            FibonacciHeap heap = new FibonacciHeap();
            for (int i = 1000; i >= 1; i--) heap.insert(i);
            for (int i = 1; i <= 1000; i++) {
                assertEquals(i, heap.min().key);
                heap.removeMin();
            }
        });
    }

    @Test
    void testShuffledInsert() {
        assertTimeoutPreemptively(Duration.ofSeconds(5), () -> {
            FibonacciHeap heap = new FibonacciHeap();
            List<Integer> nums = new ArrayList<>();
            for (int i = 1; i <= 10000; i++) nums.add(i);
            Collections.shuffle(nums);

            for (int i : nums) heap.insert(i);

            for (int i = 1; i <= 10000; i++) {
                assertEquals(i, heap.min().key);
                heap.removeMin();
            }
        });
    }

    @Test
    void testDuplicates() {
        assertTimeoutPreemptively(Duration.ofSeconds(5), () -> {
            FibonacciHeap heap = new FibonacciHeap();
            int[] nums = {3,3,3,4,4,4};

            for (int i : nums) heap.insert(i);

            for (int i : nums) {
                assertEquals(i, heap.min().key);
                heap.removeMin();
            }
        });
    }

    @Test
    void testMerge() {
        assertTimeoutPreemptively(Duration.ofSeconds(5), () -> {
            FibonacciHeap h1 = new FibonacciHeap();
            FibonacciHeap h2 = new FibonacciHeap();

            h1.insert(5);
            h1.insert(1);
            h2.insert(4);
            h2.insert(2);

            h1.merge(h2);

            assertEquals(1, h1.min().key);

            int[] expected = {1,2,4,5};
            for (int e : expected) {
                assertEquals(e, h1.min().key);
                h1.removeMin();
            }
        });
    }

    @Test
    void testDecreaseKeyNewMin() {
        assertTimeoutPreemptively(Duration.ofSeconds(5), () -> {
            FibonacciHeap heap = new FibonacciHeap();
            FibonacciHeap.Node n1 = heap.insert(10);
            heap.insert(20);

            heap.decreaseKey(n1, 1);
            assertEquals(1, heap.min().key);
        });
    }

    @Test
    void testDecreaseKeyCascade() {
        assertTimeoutPreemptively(Duration.ofSeconds(5), () -> {
            FibonacciHeap heap = new FibonacciHeap();

            FibonacciHeap.Node a = heap.insert(10);
            FibonacciHeap.Node b = heap.insert(20);
            FibonacciHeap.Node c = heap.insert(30);

            heap.removeMin();
            heap.decreaseKey(c, 5);

            assertEquals(5, heap.min().key);
        });
    }

    @Test
    void testStructureAfterOperations() {
        assertTimeoutPreemptively(Duration.ofSeconds(5), () -> {
            FibonacciHeap heap = new FibonacciHeap();
            heap.insert(3);
            heap.insert(1);
            heap.insert(2);

            FibonacciHeap.Node min = heap.min();
            assertEquals(1, min.key);

            heap.insert(0);
            assertEquals(0, heap.min().key);

            heap.removeMin();
            assertEquals(1, heap.min().key);
        });
    }
}