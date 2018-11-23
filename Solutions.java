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
