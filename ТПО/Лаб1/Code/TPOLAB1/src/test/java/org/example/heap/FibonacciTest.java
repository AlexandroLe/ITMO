package org.example.heap;

import org.junit.jupiter.api.Test;
import java.util.*;
import java.util.stream.IntStream;
import static org.junit.jupiter.api.Assertions.*;

public class FibonacciTest {

    @Test
    void testInsertIntoEmpty() {
        Fibonacci heap = new Fibonacci();
        Fibonacci.Node n = heap.insert(42);
        assertEquals(42, heap.min().key);
        verifyStructure(heap);
    }

    @Test
    void testInsertMultipleElements() {
        Fibonacci heap = new Fibonacci();
        int[] values = {5, 3, 7, 2, 8};
        for (int v : values) {
            heap.insert(v);
            verifyStructure(heap);
        }
        assertEquals(2, heap.min().key);
    }

    @Test
    void testInsertDuplicateMin() {
        Fibonacci heap = new Fibonacci();
        heap.insert(3);
        heap.insert(3);
        heap.insert(5);
        assertEquals(3, heap.min().key);
        verifyStructure(heap);
    }

    @Test
    void testInsertZeroValue() {
        Fibonacci heap = new Fibonacci();
        heap.insert(0);
        assertEquals(0, heap.min().key);
        verifyStructure(heap);
    }

    @Test
    void testReversedInsert() {
        Fibonacci heap = new Fibonacci();
        for (int i = 5; i >= 1; i--) heap.insert(i);
        assertEquals(1, heap.min().key);
        verifyStructure(heap);
    }

    @Test
    void testShuffledInsert() {
        Fibonacci heap = new Fibonacci();
        List<Integer> nums = Arrays.asList(3, 7, 1, 5, 9);
        Collections.shuffle(nums);
        for (int n : nums) heap.insert(n);
        assertEquals(1, heap.min().key);
        verifyStructure(heap);
    }

    @Test
    void testMinOnEmptyHeap() {
        Fibonacci heap = new Fibonacci();
        assertNull(heap.min());
    }

    @Test
    void testMergeWithEmpty() {
        Fibonacci h1 = new Fibonacci();
        Fibonacci h2 = new Fibonacci();
        h1.insert(5);
        h1.merge(h2);
        assertEquals(5, h1.min().key);
        verifyStructure(h1);
    }

    @Test
    void testMergeTwoEmpty() {
        Fibonacci h1 = new Fibonacci();
        Fibonacci h2 = new Fibonacci();
        h1.merge(h2);
        assertNull(h1.min());
        assertTrue(h1.isEmpty());
    }

    @Test
    void testMergeNonEmptyHeaps() {
        Fibonacci h1 = new Fibonacci();
        Fibonacci h2 = new Fibonacci();
        h1.insert(10); h1.insert(5);
        h2.insert(7); h2.insert(2);
        h1.merge(h2);
        assertEquals(2, h1.min().key);
        verifyStructure(h1);
    }

    @Test
    void testRemoveMinEmpty() {
        Fibonacci heap = new Fibonacci();
        assertNull(heap.removeMin());
    }

    @Test
    void testRemoveMinSingle() {
        Fibonacci heap = new Fibonacci();
        heap.insert(42);
        Object min = heap.removeMin();
        assertEquals(42, min);
        assertNull(heap.min());
    }

    @Test
    void testRemoveMinMultiple() {
        Fibonacci heap = new Fibonacci();
        int[] values = {4, 1, 3};
        for (int v : values) heap.insert(v);

        Object min = heap.removeMin();
        assertEquals(1, min);
        assertEquals(3, heap.min().key);
        verifyStructure(heap);

        min = heap.removeMin();
        assertEquals(3, min);
        assertEquals(4, heap.min().key);
        verifyStructure(heap);

        min = heap.removeMin();
        assertEquals(4, min);
        assertNull(heap.min());
    }

    @Test
    void testDecreaseKeyNewMin() {
        Fibonacci heap = new Fibonacci();
        Fibonacci.Node n1 = heap.insert(10);
        heap.insert(20);
        heap.decreaseKey(n1, 5);
        assertEquals(5, heap.min().key);
        verifyStructure(heap);
    }

    @Test
    void testDecreaseKeyNoMinChange() {
        Fibonacci heap = new Fibonacci();
        Fibonacci.Node n1 = heap.insert(10);
        heap.insert(5);
        heap.decreaseKey(n1, 8);
        assertEquals(5, heap.min().key);
        verifyStructure(heap);
    }

    @Test
    void testDecreaseKeyCascade() {
        Fibonacci heap = new Fibonacci();
        Fibonacci.Node n1 = heap.insert(10);
        Fibonacci.Node n2 = heap.insert(20);
        Fibonacci.Node n3 = heap.insert(30);
        heap.removeMin();
        heap.decreaseKey(n3, 5);
        assertEquals(5, heap.min().key);
        verifyStructure(heap);
    }

    @Test
    void testDecreaseKeyIllegal() {
        Fibonacci heap = new Fibonacci();
        Fibonacci.Node n1 = heap.insert(10);
        assertThrows(IllegalArgumentException.class, () -> heap.decreaseKey(n1, 15));
    }

    @Test
    void testDecreaseKeyToZero() {
        Fibonacci heap = new Fibonacci();
        Fibonacci.Node n1 = heap.insert(10);
        heap.decreaseKey(n1, 0);
        assertEquals(0, heap.min().key);
        verifyStructure(heap);
    }

    @Test
    void testDeleteElement() {
        Fibonacci heap = new Fibonacci();
        Fibonacci.Node n = heap.insert(10);
        heap.delete(n);
        assertNull(heap.min());
        verifyStructure(heap);
    }

    @Test
    void testRemoveFromRootListMultiple() {
        Fibonacci heap = new Fibonacci();
        Fibonacci.Node n1 = heap.insert(10);
        Fibonacci.Node n2 = heap.insert(5);
        heap.removeFromRootList(n1);
        assertEquals(5, heap.min().key);
        verifyStructure(heap);
    }

    @Test
    void testRemoveFromRootListSingle() {
        Fibonacci heap = new Fibonacci();
        Fibonacci.Node n1 = heap.insert(10);
        heap.removeFromRootList(n1);
        assertNull(heap.min());
    }

    @Test
    void stressInsert() {
        Fibonacci heap = new Fibonacci();
        IntStream.rangeClosed(1, 100000).forEach(heap::insert);
        assertEquals(1, heap.min().key);
    }

    @Test
    void stressRemoveMin() {
        Fibonacci heap = new Fibonacci();
        IntStream.rangeClosed(1, 100000).forEach(heap::insert);
        IntStream.rangeClosed(1, 100000).forEach(i -> {
            Fibonacci.Node currentMin = heap.min();
            if (currentMin != null) assertEquals(i, currentMin.key);
            heap.removeMin();
        });
        assertNull(heap.min());
    }

    @Test
    void stressDecreaseKey() {
        Fibonacci heap = new Fibonacci();
        List<Fibonacci.Node> nodes = new ArrayList<>();
        IntStream.rangeClosed(1, 10000).forEach(i -> nodes.add(heap.insert(i + 1000)));
        Collections.shuffle(nodes);
        for (Fibonacci.Node n : nodes) heap.decreaseKey(n, 0);
        assertEquals(0, heap.min().key);
        verifyStructure(heap);
    }

    private void verifyStructure(Fibonacci heap) {
        if (heap.min == null) return;
        Set<Fibonacci.Node> visited = new HashSet<>();
        Queue<Fibonacci.Node> queue = new LinkedList<>();
        queue.add(heap.min);
        while (!queue.isEmpty()) {
            Fibonacci.Node node = queue.poll();
            if (!visited.add(node)) continue;
            Fibonacci.Node start = node;
            Fibonacci.Node cur = node;
            do {
                if (cur.left.right != cur || cur.right.left != cur) {
                    fail("Некорректные связи left/right");
                }
                if (cur.child != null) queue.add(cur.child);
                cur = cur.right;
            } while (cur != start);
        }
    }
}