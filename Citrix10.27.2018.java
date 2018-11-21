// Suppose we have some input data describing a graph of relationships between parents and children over multiple generations. The data is formatted as a list of (parent, child) pairs, where each individual is assigned a unique integer identifier.

// For example, in this diagram, 3 is a child of 1 and 2, and 5 is a child of 4:
            
// 1   2   4
//  \ /   / \
//   3   5   8
//    \ / \   \
//     6   7   9

// Write a function that takes this data as input and returns two collections: one containing all individuals with zero known parents, and one containing all individuals with exactly one known parent.

// Sample output (pseudocode):
// [
//   [1, 2, 4],   // Individuals with zero parents
//   [5, 7, 8, 9] // Individuals with exactly one parent
// ]



import java.io.*;
import java.util.*;

/*
 * To execute Java, please define "static void main" on a class
 * named Solution.
 *
 * If you need more classes, simply define them inline.
 */

class Solution {
  public static void main(String[] args) {
    int[][] parentChildPairs = new int[][] {
        {1, 3}, {2, 3}, {3, 6}, {5, 6}, {5, 7},
        {4, 5}, {4, 8}, {8, 9}
    };
    
    int[] newone = new int[parentChildPairs.length];
    System.out.println(parentChildPairsHelper(parentChildPairs));

  }
  
  
  public static HashSet<Integer> parentChildPairsHelper(int[][] parentChildPairs) {

    int[] children = new int[parentChildPairs.length];
    int[] parents = new int[parentChildPairs.length];
    for (int i = 0; i< parentChildPairs.length; i++){
      children[i] = parentChildPairs[i][0];
      parents[i] = parentChildPairs[i][1];
    }
    
    ArrayList <Integer> temp = new ArrayList<>();
    ArrayList <Integer> result = new ArrayList<>();
    
    for(int i = 0; i< parentChildPairs.length; i++) {
      for(int j = 0; j< parentChildPairs.length; j++) {
        if (parentChildPairs[i][0] == parentChildPairs[j][1]) {
          temp.add(parentChildPairs[i][0]);
        }
      }
    }
    
    for (int i = 0; i< parentChildPairs.length; i++) {
      if(!temp.contains(parentChildPairs[i][0])) {
        result.add(parentChildPairs[i][0]);
      }
    }
    
    HashSet<Integer> newResult = new HashSet<Integer>();
    for (int i=0; i< result.size(); i++) {
      newResult.add(result.get(i));
    }
    
    // System.out.println(newResult);
    
    HashSet<Integer> newList = new HashSet<Integer>();
    HashMap<Integer, Integer> counts = new HashMap<>();
    for (int i=0; i<children.length; i++){
      if(!newResult.contains(children[i])) {
        newList.add(children[i]);
      }
        
    }
    
    System.out.println(newList);
    
    return newResult;
  }
  
  
  
}