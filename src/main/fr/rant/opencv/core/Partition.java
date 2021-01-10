package fr.rant.opencv.core;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.function.BiPredicate;

public class Partition {

    public static void main(final String[] args) {
        final List<String> list = new LinkedList<>(Arrays.asList("aa", "ab", "aa", "bb", "bb", "a", "b", "a", "b", "b"));
        final List<Integer> list2 = new ArrayList<>(Arrays.asList(1, 2, 10, 11, 10, 42, 3, 2, 3));

        final List<Integer> labels = new ArrayList<>();
        int classes = partition(list, labels);
        System.out.println("There are " + classes + " classes of equal strings:");
        System.out.println(labels);

        classes = partition(list2, labels);
        System.out.println("There are " + classes + " classes of equal integers:");
        System.out.println(labels);

        classes = partition(list, labels, (str1, str2) -> str1.length() == str2.length());
        System.out.println("There are " + classes + " classes of strings with same length:");
        System.out.println(labels);

        classes = partition(list2, labels, (i1, i2) -> Math.abs(i1 - i2) <= 1);
        System.out.println("There are " + classes + " classes of integers within 1 range:");
        System.out.println(labels);
    }

    /**
     * Overloaded method with default equality predicate
     *
     * @param vec    vector of elements to be partitioned
     * @param labels output list of labels
     * @return number of classes
     */
    public static <E> int partition(final List<E> vec, final List<Integer> labels) {
        return partition(vec, labels, E::equals);
    }

    /**
     * Port of C++ partition function
     *
     * @param inputVec  list of elements to be partitioned
     * @param labels    output list of labels
     * @param predicate predicate to test whether two elements belong to the same class
     * @return number of classes
     */
    public static <E> int partition(final List<E> inputVec, final List<Integer> labels, final BiPredicate<E, E> predicate) {

        final int n = inputVec.size();

        final ArrayList<E> vec = new ArrayList<>(inputVec);

        final int PARENT = 0;
        final int RANK = 1;

        final int[][] nodes = new int[n * 2][2];

        // The first O(n) pass: create n single-vertex trees
        for (final int[] node : nodes) {
            node[PARENT] = -1;
            node[RANK] = 0;
        }

        // The main O(n^2) pass: merge connected components
        for (int i = 0; i < n; i++) {
            int root = i;

            // find root
            while (nodes[root][PARENT] >= 0) {
                root = nodes[root][PARENT];
            }

            for (int j = 0; j < n; j++) {
                if (i == j || !predicate.test(vec.get(i), vec.get(j))) {
                    continue;
                }
                int root2 = j;

                while (nodes[root2][PARENT] >= 0) {
                    root2 = nodes[root2][PARENT];
                }

                if (root2 != root) {
                    // unite both trees
                    root = uniteTrees(nodes, i, root, j, root2);
                }
            }
        }

        // Final O(n) pass: enumerate classes
        final Integer[] _labels = new Integer[n];
        int nclasses = 0;

        for (int i = 0; i < n; i++) {
            int root = i;
            while (nodes[root][PARENT] >= 0) {
                root = nodes[root][PARENT];
            }
            // re-use the rank as the class label
            if (nodes[root][RANK] >= 0) {
                nodes[root][RANK] = ~nclasses++;
            }
            _labels[i] = ~nodes[root][RANK];
        }
        labels.clear();
        labels.addAll(Arrays.asList(_labels));
        return nclasses;
    }

    private static int uniteTrees(final int[][] nodes, final int i, int root, final int j, final int root2) {
        final int rank = nodes[root][1];
        final int rank2 = nodes[root2][1];
        if (rank > rank2) {
            nodes[root2][0] = root;
        } else {
            nodes[root][0] = root2;
            nodes[root2][1] += (rank == rank2 ? 1 : 0);
            root = root2;
        }
        assert (nodes[root][0] < 0);

        int k = j;
        int parent;

        // compress the path from node2 to root
        while ((parent = nodes[k][0]) >= 0) {
            nodes[k][0] = root;
            k = parent;
        }

        // compress the path from node to root
        k = i;
        while ((parent = nodes[k][0]) >= 0) {
            nodes[k][0] = root;
            k = parent;
        }
        return root;
    }
}