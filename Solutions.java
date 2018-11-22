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

