package org.example.heap;

import java.util.ArrayList;

public class Fibonacci {

    public static void main(String[] args) {
        Fibonacci heap = new Fibonacci();
        heap.insert(5);
        heap.insert(3);
        heap.insert(7);
        if (heap.min() != null) System.out.println(heap.min().key);
    }

    Node min = null;
    private int n = 0;

    public boolean isEmpty() {
        return min == null;
    }

    public Node insert(Object data, Comparable key) {
        Node node = new Node(data, key);
        if (min == null) {
            min = node;
        } else {
            insertIntoRootList(node);
            if (key.compareTo(min.key) < 0) min = node;
        }
        n++;
        return node;
    }

    public Node insert(int value) {
        return insert(value, value);
    }

    public Node min() {
        return min;
    }

    public Object removeMin() {
        Node z = min;
        if (z == null) return null;

        if (z.child != null) {
            Node child = z.child;
            do {
                Node next = child.right;
                insertIntoRootList(child);
                child.parent = null;
                child = next;
            } while (child != z.child);
        }

        removeFromRootList(z);

        n--;

        if (min != null) {
            consolidate();
        }

        return z.data;
    }

    public void merge(Fibonacci other) {
        if (other == null || other.min == null) return;

        if (this.min == null) {
            this.min = other.min;
            this.n = other.n;
            return;
        }

        Node thisRight = this.min.right;
        Node otherLeft = other.min.left;

        this.min.right = other.min;
        other.min.left = this.min;

        thisRight.left = otherLeft;
        otherLeft.right = thisRight;

        if (other.min.key.compareTo(this.min.key) < 0) this.min = other.min;

        this.n += other.n;
    }

    public void decreaseKey(Node x, Comparable k) {
        if (x == null) return;
        if (k.compareTo(x.key) > 0) throw new IllegalArgumentException("new key is greater");

        x.key = k;
        Node y = x.parent;

        if (y != null && x.key.compareTo(y.key) < 0) {
            cut(x, y);
            cascadingCut(y);
        }

        if (min == null) {
            min = x;
            x.left = x.right = x;
        } else if (x.key.compareTo(min.key) < 0) {
            min = x;
        }
    }

    public void delete(Node x) {
        if (x == null || min == null) return;
        decreaseKey(x, Integer.MIN_VALUE);
        removeMin();
    }

    private void consolidate() {
        if (min == null) return;
        int size = ((int) Math.floor(Math.log(n) / Math.log(2))) + 2;
        Node[] A = new Node[size];

        ArrayList<Node> roots = new ArrayList<>();
        Node x = min;
        if (x != null) {
            do {
                roots.add(x);
                x = x.right;
            } while (x != min);
        }

        for (Node w : roots) {
            x = w;
            int d = x.degree;
            while (A[d] != null) {
                Node y = A[d];
                if (x.key.compareTo(y.key) > 0) {
                    Node temp = x;
                    x = y;
                    y = temp;
                }
                link(y, x);
                A[d] = null;
                d++;
                if (d >= A.length) {
                    Node[] newA = new Node[d + 2];
                    System.arraycopy(A, 0, newA, 0, A.length);
                    A = newA;
                }
            }
            A[d] = x;
        }

        min = null;
        for (Node a : A) {
            if (a != null) {
                if (min == null) {
                    min = a;
                    a.left = a;
                    a.right = a;
                } else {
                    insertIntoRootList(a);
                    if (a.key.compareTo(min.key) < 0) min = a;
                }
            }
        }
    }

    private void link(Node y, Node x) {
        removeFromRootList(y);
        y.parent = x;
        y.mark = false;

        if (x.child == null) {
            x.child = y;
            y.left = y;
            y.right = y;
        } else {
            y.left = x.child;
            y.right = x.child.right;
            x.child.right.left = y;
            x.child.right = y;
        }

        x.degree++;
    }

    private void cut(Node x, Node y) {
        if (x == null || y == null) return;

        if (x.right == x) {
            y.child = null;
        } else {
            x.right.left = x.left;
            x.left.right = x.right;
            if (y.child == x) y.child = x.right;
        }

        y.degree--;
        insertIntoRootList(x);
        x.parent = null;
        x.mark = false;
    }

    private void cascadingCut(Node y) {
        if (y == null) return;
        Node z = y.parent;
        if (z != null) {
            if (!y.mark) y.mark = true;
            else {
                cut(y, z);
                cascadingCut(z);
            }
        }
    }

    private void insertIntoRootList(Node node) {
        if (min == null) return;
        node.left = min;
        node.right = min.right;
        min.right.left = node;
        min.right = node;
    }

    public void removeFromRootList(Node node) {
        if (node == null || min == null) return;

        if (node.right == node) {
            min = null;
        } else {
            node.left.right = node.right;
            node.right.left = node.left;
            if (min == node) min = node.right;
        }
        node.left = node.right = node;
    }

    static class Node {
        Object data;
        Comparable key;
        Node parent;
        Node child;
        Node left;
        Node right;
        int degree;
        boolean mark;

        Node(Object data, Comparable key) {
            this.data = data;
            this.key = key;
            this.left = this;
            this.right = this;
        }
    }
}