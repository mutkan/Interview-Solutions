//#1 Two Sum
class Solution {
    public int[] twoSum(int[] nums, int target) {
        HashMap <Integer, Integer> map= new HashMap<>();
        for(int i=0; i<nums.length; i++) {
            if(map.containsKey(target-nums[i])) {
                return new int[] {i, map.get(target-nums[i])};
            }
            map.put(nums[i], i);
        }
        throw new IllegalArgumentException("No Solution");
    }
}


//#20 Valid Parentheses
class Solution {
    private HashMap<Character, Character> mappings;
    
    public Solution() {
        this.mappings = new HashMap<Character, Character>();
        this.mappings.put(')', '(');
        this.mappings.put(']', '[');
        this.mappings.put('}', '{');
    }
    
    public boolean isValid(String s) {
        Stack<Character> stack = new Stack<>();
        for(int i=0; i<s.length(); i++) {
            char c = s.charAt(i);
            if(this.mappings.containsKey(c)) {
                char top = stack.empty() ? '#':stack.pop();
                if(top != this.mappings.get(c)) return false;
            } else {
                stack.push(c);
            }
        }
        return stack.isEmpty();
    }
}


//#2 Add two numbers
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution {
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        int carrybit = 0;
        ListNode dummy = new ListNode(0);
        ListNode current = dummy;
        ListNode p = l1, q = l2;
        while(p!=null | q!=null) {
            int x = (p!=null) ? p.val:0;
            int y = (q!=null) ? q.val:0;
            int sum = carrybit + x + y;
            carrybit = sum / 10;
            current.next = new ListNode(sum%10);
            current = current.next;
            if (p!=null) p = p.next;
            if (q!=null) q = q.next;
        }
        if(carrybit>0) {
            current.next = new ListNode(carrybit);
        }
        return dummy.next;
    }
}


//#3 Longest substring without repeating characters
class Solution {
    public int lengthOfLongestSubstring(String s) {
        int ans = 0;
        Map<Character, Integer> map = new HashMap<>();
        for(int i=0, j=0; j<s.length(); j++){
            if(map.containsKey(s.charAt(j))) {
                i = Math.max(map.get(s.charAt(j)), i);
            }
            ans = Math.max(ans, j-i+1);
            map.put(s.charAt(j), j+1);
        }
        return ans;
    }
}


//#206 Reverse Linked List
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution {
    public ListNode reverseList(ListNode head) {
        ListNode prev = null;
        ListNode current = head;
        while(current != null) {
            ListNode tempNext = current.next;
            current.next = prev;
            prev = current;
            current = tempNext;
        }
        return prev;
    }
}


//#7 Reverse Integer
class Solution {
    public int reverse(int x) {
        int rev = 0;
        while (x != 0) {
            int pop = x % 10;
            x /= 10;
            if((rev > Integer.MAX_VALUE/10) || (rev == Integer.MAX_VALUE/10 && pop > 7)) {
                return 0;
            }
            if((rev < Integer.MIN_VALUE/10) || (rev == Integer.MIN_VALUE/10 && pop < -8)) {
                return 0;
            }
            rev = rev * 10 + pop;
        }
        return rev;
    }
}


//#203 Remove Linked List elements
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution {
    public ListNode removeElements(ListNode head, int val) {
        if (head == null) {
            return null;
        } 
        head.next = removeElements(head.next, val);
        if (head.val == val) {
            return head.next;
        }else{
            return head;
        }
    }
}


//#146 LRU Cache
class Node {
    int key, val;
    Node pre, next;

    Node(int key, int val) {
        this.key = key;
        this.val = val;
    }
}

class LRUCache{
    int capacity;
    int count;
    Node head, tail;

    Map<Integer, Node> map;

    public LRUCache(int capacity) {
        this.capacity = capacity;
        this.count = 0;

        head = new Node(0,0);
        tail = new Node(0,0);

        head.pre = null;
        head.next = tail;
        tail.pre = head;
        tail.next = null;

        map = new HashMap<>();
    }

    private Node removeNode(Node node){
        node.pre.next = node.next;
        node.next.pre = node.pre;
        map.remove(node.key);
        count--;
        return node;
    }

    private void addNode(Node node) {
        node.next = head.next;
        node.pre = head;
        head.next = node;
        node.next.pre = node;
        map.put(node.key, node);
        count++;
    }

    private void removeLastNode() {
        map.remove(tail.pre.key);
        tail.pre = tail.pre.pre;
        tail.pre.next = tail;
        count--;
    }

    public int get(int key) {
        if(map.containsKey(key)) {
            Node node = map.get(key);
            addNode(removeNode(node));
            return node.val;
        }
        return -1;
    }

    public void put(int key, int value) {
        Node node = new Node(key, value);
        if(map.containsKey(key)) {
            removeNode(map.get(key));
        }
        if(count>=capacity) {
            removeLastNode();
        }
        addNode(node);
    }
}
/**
 * Your LRUCache object will be instantiated and called as such:
 * LRUCache obj = new LRUCache(capacity);
 * int param_1 = obj.get(key);
 * obj.put(key,value);
 */


//#344 Reverse String
class Solution1 {
    public String reverseString(String s) {
        return new StringBuilder(s).reverse().toString();
    }
}

class Solution2 {
    public String reverseString(String s) {
        int i = 0;
        int j = s.length()-1;
        byte[] words = s.getBytes();
        
        while(i<j) {
            byte temp = words[i];
            words[i] = words[j];
            words[j] = temp;
            i++;
            j--;
        }  
        return new String(words);
    }
}


//#402 Remove K digits
class Solution {
    public String removeKdigits(String num, int k) {
        int newLength = num.length() - k;
        char[] stack = new char[num.length()];
        int top = 0;
        for(int i =0; i<num.length(); i++) {
            while(top>0 && stack[top-1]>num.charAt(i) && k>0) {
                top--;
                k--;
            }
            stack[top++] = num.charAt(i);
        }
        
        int offset = 0;
        while(offset<newLength && stack[offset]=='0') {
            offset++;
        }
        return offset == newLength ? "0":new String(stack, offset, newLength-offset);
    }
}


//#19 Remove Nth of Node from end of list
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution1 {
    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        int length = 0;
        ListNode first = head;
        while(first != null) {
            first = first.next;
            length++;
        }
        length = length - n;
        first = dummy;
        while(length>0) {
            first = first.next;
            length--;
        }
        first.next = first.next.next;
        return dummy.next;
    }
}

class Solution2 {
    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode first = dummy;
        ListNode second = dummy;
        for(int i =0; i<n+1; i++) {
            second = second.next;
        }
        while(second != null) {
            first = first.next;
            second = second.next;
        }
        first.next = first.next.next;
        return dummy.next;
    }
}


//#929 Unique email addresses
class Solution {
    public int numUniqueEmails(String[] emails) {
        Set<String> str = new HashSet();
        for(String email: emails) {
            int i = email.indexOf('@');
            String local = email.substring(0, i);
            String rest = email.substring(i);
            if(local.contains("+")) {
                local = local.substring(0, local.indexOf('+'));
            }
            local = local.replaceAll(".", "");
            str.add(local+rest);
        }
        return str.size();
    }
}


//#9 Palindrome Number
class Solution {
    public boolean isPalindrome(int x) {
        if(x<0 || (x%10==0 && x!=0)) return false;
        int revert = 0;
        while(x>revert) {
            revert = revert*10 + x%10;
            x /= 10;
        }
        return x==revert || x==revert/10;
    }
}


//#21 Merge Two Sorted Lists
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution1 {
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        if(l1==null) {
            return l2;
        } else if(l2==null) {
            return l1;
        } else if(l1.val < l2.val) {
            l1.next = mergeTwoLists(l1.next, l2);
            return l1;
        } else {
            l2.next = mergeTwoLists(l1, l2.next);
            return l2;
        }
    }
}

class Solution2 {
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        ListNode pre = new ListNode(-1);
        ListNode prev = pre;
        while(l1!=null && l2!=null) {
            if(l1.val <= l2.val) {
                prev.next = l1;
                l1 = l1.next;  
            } else {
                prev.next = l2;
                l2 = l2.next;
            }
            prev = prev.next;
        }
        prev.next = l2 == null ? l1 : l2;
        return pre.next;
    }
}


//#15 3 Sum
class Solution {
    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        Arrays.sort(nums);
        for(int i=0; i<nums.length-2; i++) {
            if(i>0 && nums[i]==nums[i-1]) {
                continue;
            }
            int j = i + 1; 
            int k = nums.length - 1;
            int target = -nums[i]; //sum 0 
            while(j<k) {
                if(nums[j]+nums[k] == target) {
                    result.add(Arrays.asList(nums[i], nums[j], nums[k]));
                    j++;
                    k--;
                    while(j<k && nums[j]==nums[j-1]) j++;
                    while(j<k && nums[k]==nums[k+1]) k--;
                } else if(nums[j]+nums[k] > target) {
                    k--;
                } else {
                    j++;
                }
            }
        }
        return result;
    }
}


//#27 Remove Elements
class Solution {
    public int removeElement(int[] nums, int val) {
        int count = 0;
        for(int i=0; i<nums.length; i++) {
            if(nums[i]!=val) {
                nums[count] = nums[i];
                count++;
            }
        }
        return count;
    }
}


//#26 Remove Duplicates from sorted array
class Solution {
    public int removeDuplicates(int[] nums) {
        if(nums.length==0) return 0;
        int i = 0;
        for(int j=1; j<nums.length; j++) {
            if(nums[j]!=nums[i]) {
                i++;
                nums[i] = nums[j];
            }
        }
        return i+1;
    }
}


//#91 Decode Ways
class Solution {
    public int numDecodings(String s) {
        if(s.length() == 0) return 0;
        int[] memo = new int[s.length()+1];
        memo[s.length()]  = 1;
        memo[s.length()-1] = s.charAt(s.length()-1) != '0' ? 1 : 0;
        
        for (int i = s.length() - 2; i >= 0; i--)
            if (s.charAt(i) == '0') continue;
            else memo[i] = (Integer.parseInt(s.substring(i,i+2))<=26) ? memo[i+1]+memo[i+2] : memo[i+1];
        
        return memo[0];
    }
}


//#53 Maximum Subarray
class Solution {
    public int maxSubArray(int[] nums) {
        int maxSum = nums[0];
        int maxEnd = nums[0];
        for(int i=1; i<nums.length; i++) {
            maxSum = Math.max(maxSum+nums[i], nums[i]);
            maxEnd = Math.max(maxSum, maxEnd);
        }
        return maxEnd;
    }
}


//#125 Valid Palindrome
class Solution {
    public boolean isPalindrome(String s) {
        int start = 0;
        int end = s.length() - 1;
        while (start <= end) {
            while (start <= end && !Character.isLetterOrDigit(s.charAt(start))) {
                start++;
            }   
            while (start <= end && !Character.isLetterOrDigit(s.charAt(end))) {
                end--;
            }
            
            if (start <= end && Character.toLowerCase(s.charAt(start)) != Character.toLowerCase(s.charAt(end))) {
                return false;
            }
            start++;
            end--;
        }
        return true;
    }
}


// #215 Kth Largest Element in the Array
// A smiliar problem starting with order statistics
// I just watch an hour of two people implementing a sorting algorithm at 3:30 am? I have no life.
// https://www.youtube.com/watch?v=D35llNtkCps

// QuickSelect has the average runtime O(N)
class Solution {
    int[] nums;
    
    public int findKthLargest(int[] nums, int k) {
        this.nums = nums;
        int size = nums.length;
        return quickselect(0, size-1, size-k);
    }
    
    public void swap(int a, int b) {
        int temp = this.nums[a];
        this.nums[a] = this.nums[b];
        this.nums[b] = temp;
    }
    
    public int partition(int left, int right, int pivot_index) {
        int pivot = this.nums[pivot_index];
        // 1. move pivot to end
        swap(pivot_index, right);
        int store_index = left;
        
        // 2. move all smaller elements to the left
        for (int i=left; i<=right; i++) {
            if (this.nums[i] < pivot) {
                swap(store_index, i);
                store_index++;
            }
        }
        
        // 3. move pivot to its final place
        swap(store_index, right);
        
        return store_index;
    }
    
    public int quickselect(int left, int right, int k_smallest) {
        // return k-th smallest element of list
        if (left == right)  return this.nums[left];
        
        Random random_num = new Random();
        int pivot_index = left + random_num.nextInt(right - left);
        
        pivot_index = partition(left, right, pivot_index);
        
        //pivot is on (N-k)th smallest place
        if (k_smallest == pivot_index) {
            return this.nums[k_smallest];
        } else if (k_smallest < pivot_index) {
            //left side
            return quickselect(left, pivot_index-1, k_smallest);
        } else {
            return quickselect(pivot_index+1, right, k_smallest);
        }     
        
    }
}


// #441 Arranging Coins
class Solution1 {
    public int arrangeCoins(int n) {
        int current = 1, remainder = n-1;
        while (remainder >= current+1) {
            current++;
            remainder -= current;
        }
        return n == 0 ? 0 : current;
    }
}

class Solution2 {
    public int arrangeCoins(int n) {
        // root of n = (1+x)*x/2
        return (int)((-1 + Math.sqrt(1+8*(long)n))/2);
    }
}


// #36 Valid Sudoku
class Solution {
    public boolean isValidSudoku(char[][] board) {
        Set seen = new HashSet();
        for (int i=0; i<9; ++i) {
            for (int j=0; j<9; ++j) {
                char number = board[i][j];
                if (number != '.')
                    if (!seen.add(number + " in row " + i) ||
                        !seen.add(number + " in column " + j) ||
                        !seen.add(number + " in block " + i/3 + "-" + j/3))
                        return false;
            }
        }
        return true;
    }
}


// #14 Longest Common Prefix
class Solution1 {
    public String longestCommonPrefix(String[] strs) {
        String result = "";
        int min_length = Integer.MAX_VALUE;
        if (strs.length == 0)
            return result;
        if (strs.length == 1) 
            return strs[0];
        for (int i=0; i<strs.length; i++) {
            if (strs[i].length() < min_length)
                min_length = strs[i].length();
        }
        
        for (int i=0; i<min_length; i++) {
            char c = strs[0].charAt(i);
            for (int j=1; j<strs.length; j++) {
                if (c != strs[j].charAt(i))
                    return result;
            }
            result += c;
        }
        return result;
    }
}

class Solution2 {
    public String longestCommonPrefix(String[] strs) {
        // horizontal scanning
        if (strs.length == 0) return "";
        String prefix = strs[0];
        for (int i=0; i<strs.length; i++) {
            while (strs[i].indexOf(prefix) != 0) {
                prefix = prefix.substring(0, prefix.length()-1);
                if (prefix.isEmpty()) return "";
            }
        }
        return prefix;
    }
}

class Solution3 {
    public String longestCommonPrefix(String[] strs) {
        // verical scanning
        if (strs == null || strs.length == 0) return "";
        for (int i=0; i<strs[0].length(); i++) {
            char c = strs[0].charAt(i);
            for (int j=1; j<strs.length; j++) {
                if (i == strs[j].length() || strs[j].charAt(i) != c) {
                    return strs[0].substring(0, i);
                }
            }
        }
        return strs[0];
    }
}

class Solution4 {
    public String longestCommonPrefix(String[] strs) {
        // binary search
        if (strs == null || strs.length == 0)
            return "";
        int minLen = Integer.MAX_VALUE;
        for (String str : strs)
            minLen = Math.min(minLen, str.length());
        int low = 1;
        int high = minLen;
        while (low <= high) {
            int middle = (low + high) / 2;
            if (isCommonPrefix(strs, middle))
                low = middle + 1;
            else
                high = middle - 1;
        }
        return strs[0].substring(0, (low + high) / 2);
    }

    private boolean isCommonPrefix(String[] strs, int len){
        String str1 = strs[0].substring(0,len);
        for (int i = 1; i < strs.length; i++)
            if (!strs[i].startsWith(str1))
                return false;
        return true;
    }
}


// #686 Repeated String Match
class Solution {
    public int repeatedStringMatch(String A, String B) {
        StringBuilder str = new StringBuilder(A);
        int i;
        for (i=1; str.length()<B.length(); i++) {
            str.append(A);      
        }
        if (str.indexOf(B) >= 0) return i;
        if (str.append(A).indexOf(B) >= 0)  return i+1;
        return -1;
        //When k = q+1, A * k is already big enough to try all positions for B
    }
}

class Solution2 {
    // Rabin-Karp
    import java.math.BigInteger;

    public boolean check(int index, String A, String B) {
        for (int i = 0; i < B.length(); i++) {
            if (A.charAt((i + index) % A.length()) != B.charAt(i)) {
                return false;
            }
        }
        return true;
    }
    public int repeatedStringMatch(String A, String B) {
        int q = (B.length() - 1) / A.length() + 1;
        int p = 113, MOD = 1_000_000_007;
        int pInv = BigInteger.valueOf(p).modInverse(BigInteger.valueOf(MOD)).intValue();

        long bHash = 0, power = 1;
        for (int i = 0; i < B.length(); i++) {
            bHash += power * B.codePointAt(i);
            bHash %= MOD;
            power = (power * p) % MOD;
        }

        long aHash = 0; power = 1;
        for (int i = 0; i < B.length(); i++) {
            aHash += power * A.codePointAt(i % A.length());
            aHash %= MOD;
            power = (power * p) % MOD;
        }

        if (aHash == bHash && check(0, A, B)) return q;
        power = (power * pInv) % MOD;

        for (int i = B.length(); i < (q + 1) * A.length(); i++) {
            aHash -= A.codePointAt((i - B.length()) % A.length());
            aHash *= pInv;
            aHash += power * A.codePointAt(i % A.length());
            aHash %= MOD;
            if (aHash == bHash && check(i - B.length() + 1, A, B)) {
                return i < q * A.length() ? q : q + 1;
            }
        }
        return -1;
    }
}


// #295 Find Median from Data Stream
class MedianFinder {
    /** initialize your data structure here. */
    PriorityQueue<Integer> min = new PriorityQueue();
    PriorityQueue<Integer> max = new PriorityQueue(Collections.reverseOrder());
    
    public void addNum(int num) {
        max.offer(num);
        min.offer(max.poll());
        if (max.size() < min.size()) {
            max.offer(min.poll());
        }      
    }
    
    public double findMedian() {
        if (max.size() == min.size()) {
            return ((double)max.peek() + (double)min.peek()) / 2;
        } else {
            return max.peek();
        }
    }
}


// #50 Pow(x,n)
class Solution1 {
    public double myPow(double x, int n) {
        long N = n;
        if (N < 0) {
            x = 1/x;
            N = -N;
        }
        return fastPower(x,N);
    }
    
    private double fastPower(double x, long n) {
        if (n == 0) {
            return 1.0;
        }
        double half = fastPower(x, n/2);
        if (n % 2 == 0) {
            return half * half;
        } else {
            return half * half * x;
        }
    }
}

class Solution2 {
    public double myPow(double x, int n) {
        long N = n;
        if (N < 0) {
            x = 1 / x;
            N = -N;
        }
        double ans = 1;
        double current_product = x;
        for (long i = N; i > 0; i /= 2) {
            if ((i % 2) == 1) {
                ans = ans * current_product;
            }
            current_product = current_product * current_product;
        }
        return ans;
    }
}


// #69 Sqrt(x)
class Solution {
    public int mySqrt(int x) {
        if (x == 0) return 0;
        long y = x;
        while (y > x/y) {
            // Newton's method
            y = (y + x/y)/2;
        }
        return (int)y;
    }
}


// #141 Linked List Cycle
/**
 * Definition for singly-linked list.
 * class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) {
 *         val = x;
 *         next = null;
 *     }
 * }
 */
public class Solution {
    public boolean hasCycle(ListNode head) {
        if (head == null || head.next == null)  return false;
        ListNode fast = head.next;
        ListNode slow = head;
        while (fast != slow) {
            if (fast == null || fast.next == null)  return false;
            fast = fast.next.next;
            slow = slow.next;
        }
        return true;
    }
}


// #142 Linked List Cycle II
/**
 * Definition for singly-linked list.
 * class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) {
 *         val = x;
 *         next = null;
 *     }
 * }
 */
public class Solution1 {
    public ListNode detectCycle(ListNode head) {
        Set<ListNode> visited = new HashSet<>();
        ListNode current = head;
        while (current != null) {
            if (visited.contains(current)) {
                return current;
            }
            visited.add(current);
            current = current.next;
        }
        return null;
    }
}

public class Solution2 {
    public ListNode detectCycle(ListNode head) {
        if (head == null || head.next == null)  return null;
        ListNode intersect = findCycle(head);
        if (intersect == null) return null;
        
        ListNode first = head;
        ListNode second = intersect;
        while (first != second) {
            first = first.next;
            second = second.next;
        }
        return first;
    }
    
    public ListNode findCycle(ListNode head) {
        ListNode fast = head;
        ListNode slow = head;
        
        while (fast != null && fast.next != null) {
            fast = fast.next.next;
            slow = slow.next;
            if (slow == fast)   return slow;
        }
        return null;
    }
}


// #121 Best time to buy and sell stock
class Solution1 {
    public int maxProfit(int[] prices) {
        int maxProfit = 0;
        for (int i = 0; i < prices.length; i++) {
            for (int j = i + 1; j < prices.length; j++) {
                int profit = prices[j] - prices[i];
                if (profit > maxProfit) {
                    maxProfit = profit;
                }
            }
        }
        return maxProfit;
    }
}

class Solution2 {
    public int maxProfit(int[] prices) {
        int minPrice = Integer.MAX_VALUE;
        int maxProfit = 0;
        for (int i = 0; i < prices.length; i++) {
            if (prices[i] < minPrice) {
                minPrice = prices[i];
            } else if (prices[i] - minPrice > maxProfit) {
                maxProfit = prices[i] - minPrice;
            }
        }
        return maxProfit;
    }
}


// #122 Best time to buy and sell stock II
class Solution {
    public int maxProfit(int[] prices) {
        int i = 0;
        if (prices.length < 2) return 0;
        int valley = prices[0];
        int peak = prices[0];
        int maxProfit = 0;
        while (i < prices.length - 1) {
            while (i < prices.length - 1 && prices[i] >= prices[i+1]) {
                i++;
            }
            valley = prices[i];
            while (i < prices.length - 1 && prices[i] <= prices[i+1]) {
                i++;
            }
            peak = prices[i];
            maxProfit += peak - valley;
        }
        return maxProfit;
    }
}


// #79 Word Search
class Solution {
    public boolean exist(char[][] board, String word) {
        int m = board.length;
        int n = board[0].length;
        //这个数组记录了每个点是否使用过。这里不同于N皇后问题中只用了三个set，因为这里要精确到点，而不是整条边。
        boolean[][] used = new boolean[m][n];
        //得到word的字符数组，在之后的dfs过程中也是一直传递这个数组
        char[] wordChars = word.toCharArray(); 
        //对于每个点分别调用dfs,只有头字符相同时才调用，可以减少时间
        for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
          if (wordChars[0] == board[i][j]) {
            boolean res = dfs(board, wordChars, used, i, j, 0);
              if (res)  return true;
            }
          }
        }    
        return false;
    }

    public boolean dfs(char[][] board, char[] word, boolean[][] used, int x, int y, int level) {
      //截止到此层之前都没有被终止（层数等于word长度），证明可以找到匹配的单词
      if (level == word.length) {
        return true;
      }
      //判断是否当前要判断的char根本不存在board中
      if (x >= board.length || x < 0 || y >= board[0].length || y < 0) {
        return false;
      }
      //当前char与当前word对应层的char不相同则直接返回
      if (word[level] != board[x][y]) {
        return false;
      }
      //如果当前元素被使用过了，则直接返回
      if (used[x][y] == true) {
        return false;
      } 
      //表记当前元素为使用过
      used[x][y] = true;   
      //注意这是有返回值情况下dfs的标准写法，dfs递归调用，只要有一个是true, 则最终结果是true!
      boolean result = dfs(board, word, used, x + 1, y, level + 1)
        || dfs(board, word, used, x, y + 1, level + 1)
        || dfs(board, word, used, x - 1, y, level + 1)
        || dfs(board, word, used, x, y - 1, level + 1);
      //恢复现场，删除标记
      used[x][y] = false;
      return result;
    }
}


// #5 Longest Palindromic Substring
class Solution {
    public String longestPalindrome(String s) {
        // expand around center, check (2n-1) centers
        if (s == null || s.length() < 1)    return "";
        int start = 0, end = 0;
        for (int i = 0; i < s.length(); i++) {
            int len1 = expandAroundCenter(s, i, i);
            int len2 = expandAroundCenter(s, i, i+1);
            int len = Math.max(len1, len2);
            if (len > end - start) {
                start = i - (len - 1)/2;
                end = i + len/2;
            }
        }
        return s.substring(start, end + 1);
    }
    
    private int expandAroundCenter(String s, int left, int right) {
        int L = left, R = right;
        while (L >= 0 && R < s.length() && s.charAt(L) == s.charAt(R)) {
            L--;
            R++;
        }
        return R-L-1;
    }
}


// #977 Squares of a Sorted Array
class Solution1 {
    public int[] sortedSquares(int[] A) {
        int[] ans = new int[A.length];
        for (int i = 0; i < A.length; i++) {
            ans[i] = A[i]*A[i];
        }
        Arrays.sort(ans);
        return ans;
    }
}

class Solution2 {
    public int[] sortedSquares(int[] A) {
        int j = 0;
        while (j < A.length && A[j] < 0) {
            j++;
        }
        int i = j-1;
        
        int[] ans = new int[A.length];
        int t = 0;
        
        while (i >= 0 && j < A.length) {
            if (A[i]*A[i] < A[j]*A[j]) {
                ans[t++] = A[i]*A[i];
                i--;
            } else {
                ans[t++] = A[j]*A[j];
                j++;
            }
        }      
        while (i >= 0) {
            ans[t++] = A[i]*A[i];
            i--;
        }       
        while (j < A.length) {
            ans[t++] = A[j]*A[j];
            j++;
        }      
        return ans;
    }
}


// #38 Count and Say
class Solution {
    public String countAndSay(int n) {
        String result= "1";
        for (int i = 1; i < n; i++) {
            result = countNumbers(result);
        }
        return result;
    }
    
    private String countNumbers(String str) {
        StringBuilder s = new StringBuilder();
        int count = 1;
        char c = str.charAt(0);
        for (int i = 1; i < str.length(); i++) {
            if (str.charAt(i) == c) {
                count++;
            } else {
                s.append(count);
                s.append(c);
                c = str.charAt(i);
                count = 1;
            }
        }
        s.append(count);
        s.append(c);
        return s.toString();
    }
}


// #984 String Without AAA or BBB
class Solution {
    public String strWithout3a3b(int A, int B) {
        StringBuilder result = new StringBuilder();
        int size = A + B;
        int a = 0, b = 0;
        for (int i = 0; i < size; i++) {
            if ((A >= B && a != 2) || b == 2) {
                result.append("a");
                A--;
                a++;
                b = 0;
            } else if((B >= A && b != 2) || a ==2) {
                result.append("b");
                b++;
                B--;
                a = 0;
            }
        }
        return result.toString();
    }
}


// #149 Max Points on a Line
/**
 * Definition for a point.
 * class Point {
 *     int x;
 *     int y;
 *     Point() { x = 0; y = 0; }
 *     Point(int a, int b) { x = a; y = b; }
 * }
 */
import javafx.util.Pair; 
class Solution {
    
    Point [] points;
    int n;
    Map<Pair<Double, Double>, Integer> lines = new HashMap<Pair<Double, Double>, Integer>();
    Map<Integer, Integer> horizontal_lines = new HashMap<Integer, Integer>();
    
    public Pair<Integer, Integer> add_line(int i, int j, int count, int duplicates) {
        /*  Add a line passing through i and j points,
            Update max number of points on a line containing point i,
            Update a number of duplicates of i point
        */
        // rewrite points as coordinates
        int x1 = points[i].x;
        int x2 = points[j].x;
        int y1 = points[i].y;
        int y2 = points[j].y;
        // add a horizontal line y=const
        if ((x1 == x2) && (y1 == y2)) {
            duplicates++;
        } else if (y1 == y2) {
            horizontal_lines.put(y1, horizontal_lines.getOrDefault(y1, 1)+1);
            count = Math.max(horizontal_lines.get(y1), count);
        } else {
            // Add 0.0 to avoid "-0.0"
            double slope = 1.0 * (x1 -x2) / (y1 - y2) + 0.0;
            double c = 1.0 * (y1 * x2 - y2 * x1) / (y1 - y2) + 0.0;
            Pair p = new Pair(slope, c);
            lines.put(p, lines.getOrDefault(p, 1) + 1);
            count = Math.max(lines.get(p), count);
        }
        return new Pair(count, duplicates);
    }
    
    public int max_on_a_line_containing_point_i(int i) {
        // Compute max number of points for a line containing point i
        // inital lines passing through point i
        lines.clear();
        horizontal_lines.clear();
        int count = 1;
        int duplicates = 0;
        
        /*  Compute lines passing through point i and point j
            Update in a loopp the number of points on a line and number of duplicates of point i
        */
        for (int j = i + 1; j < n; j++) {
            Pair<Integer, Integer> p = add_line(i, j, count, duplicates);
            count = p.getKey();
            duplicates = p.getValue();
        }
        return count + duplicates;
    }
    
    public int maxPoints(Point[] points) {
        this.points = points;
        n = points.length;
        // If number of points is less than 3, they are all on the same line
        if (n < 3) {
            return n;
        }
        
        int max_count = 1;
        // Compute in a loop a max number of points on a line containing point i
        for (int i = 0; i < n - 1; i++) {
            max_count = Math.max(max_on_a_line_containing_point_i(i), max_count);
        }
        return max_count;
    }
}


// #4 Median of Two Sorted Array
class Solution {
    public double findMedianSortedArrays(int[] A, int[] B) {
        int m = A.length;
        int n = B.length;
        if (m > n) {
            int[] temp = A; A = B; B = temp;
            int tmp = m; m = n; n = tmp;
        }
        
        int iMin = 0, iMax = m, halfLen = (m + n + 1)/2;
        while (iMin <= iMax) {
            int i = (iMin + iMax)/2;
            int j = halfLen - i;
            if (i < iMax && B[j-1] > A[i]) {
                iMin = i + 1;
            } else if (i > iMin && A[i-1] > B[j]) {
                iMax = i - 1;
            } else {
                int maxLeft = 0;
                if (i == 0) {
                    maxLeft = B[j-1];
                } else if (j == 0) {
                    maxLeft = A[i-1];
                } else {
                    maxLeft = Math.max(A[i-1], B[j-1]);
                }
                if ((m+n)%2 == 1)   return maxLeft;
                
                int minRight = 0;
                if (i == m) {
                    minRight = B[j];
                } else if (j == n) {
                    minRight = A[i];
                } else {
                    minRight = Math.min(B[j], A[i]);
                }
                
                return (maxLeft + minRight) / 2.0;
            }
        }
        return 0.0;
    }
}
