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

