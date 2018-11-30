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
