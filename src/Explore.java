import ds.ListNode;
import ds.TreeNode;

import java.util.*;

public class Explore {

    /* ======================================
                        初级算法
       ====================================== */

    /* ---------
          数组
       --------- */

    // 删除排序数组中的重复项
    public int removeDuplicates(int[] nums) {
        if (nums.length == 0 || nums.length == 1) {
            return nums.length;
        }
        int len = 1;
        for (int i = 1; i < nums.length; i++) {
            if (nums[i] != nums[i - 1]) {
                nums[len] = nums[i];
                len++;
            }
        }
        return len;
    }

    // 买卖股票的最佳时机2
    public int maxProfit(int[] prices) {
        int sum = 0;
        for (int i = 1; i < prices.length; i++) {
            sum += Math.max(prices[i] - prices[i - 1], 0);
        }
        return sum;
    }

    // 旋转数组
    public void rotate(int[] nums, int k) {
        k %= nums.length;
        reverse(nums, 0, nums.length - 1);
        reverse(nums, 0, k - 1);
        reverse(nums, k, nums.length - 1);
    }
    void reverse(int[] nums, int start, int end) {
        while (start < end) {
            nums[start] ^= nums[end];
            nums[end] ^= nums[start];
            nums[start] ^= nums[end];
            start++;
            end--;
        }
    }

    // 存在重复元素
    public boolean containsDuplicate(int[] nums) {
        Set<Integer> hashSet = new HashSet<>();
        for (int num : nums) {
            if(!hashSet.add(num)) {
                return true;
            }
        }
        return false;
    }

    // 只出现一次的数字
    public int singleNumber(int[] nums) {
        int num = 0;
        for (int value : nums) {
            num ^= value;
        }
        return num;
    }

    // 两个数字的交集2
    public int[] intersect(int[] nums1, int[] nums2) {
        if (nums1.length > nums2.length) {
            return intersect(nums2, nums1);
        }
        Map<Integer, Integer> map = new HashMap<>();
        for (int i : nums1) {
            int count = map.getOrDefault(i, 0) + 1;
            map.put(i, count);
        }
        int[] ret = new int[nums1.length];
        int index = 0;
        for (int i : nums2) {
            int count = map.getOrDefault(i, 0);
            if (count > 0) {
                ret[index++] = i;
                count--;
                if (count > 0) {
                    map.put(i, count);
                } else {
                    map.remove(i);
                }
            }
        }
        return Arrays.copyOfRange(ret, 0, index);
    }

    // 加一
    public int[] plusOne(int[] digits) {
        for (int i = digits.length - 1; i >= 0; i--) {
            if (digits[i] + 1 == 10) {
                digits[i] = 0;
            } else {
                digits[i] += 1;
                return digits;
            }
        }
        if (digits[0] == 0) {
            int[] ret = new int[digits.length + 1];
            ret[0] = 1;
            return ret;
        }
        return digits;
    }

    // 移动零
    public void moveZeroes(int[] nums) {
        int index = 0;
        for (int num : nums) {
            if (num != 0) {
                nums[index++] = num;
            }
        }
        Arrays.fill(nums, index, nums.length, 0);
    }

    // 两数之和
    public int[] twoSum(int[] nums, int target) {
        int[] ret = new int[2];
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            int num = target - nums[i];
            if (map.containsKey(num)) {
                ret[0] = map.get(num);
                ret[1] = i;
                return ret;
            }
            map.put(nums[i], i);
        }
        return ret;
    }

    // 有效的数独
    public boolean isValidSudoku(char[][] board) {
        Set<Character> rowSet = new HashSet<>();
        Set<Character> colSet = new HashSet<>();
        HashSet[] areaSet = new HashSet[9];
        for (int i = 0; i < 9; i++) {
            areaSet[i] = new HashSet<>();
        }
        for (int i = 0; i < 9; i++) {
            rowSet.clear();
            colSet.clear();
            for (int j = 0; j < 9; j++) {
                char rowChar = board[i][j];
                char colChar = board[j][i];
                int areaIndex = (i / 3) * 3 + j / 3;
                if (rowChar != '.' && !rowSet.add(rowChar)) {
                    return false;
                }
                if (colChar != '.' &&!colSet.add(colChar)) {
                    return false;
                }
                if (rowChar != '.' && !areaSet[areaIndex].add(rowChar)) {
                    return false;
                }
            }
        }
        return true;
    }

    // 旋转图像
    public void rotate(int[][] matrix) {
        int len = matrix.length;
        for (int i = 0; i < len / 2 + 1; i++) {
            for (int j = i; j < len - 1 - i; j++) {
                int x, y;
                x = j;
                y = len - 1 - i;
                swap(matrix, i, j, x, y);
                x = y;
                y = len - 1 - j;
                swap(matrix, i, j, x, y);
                x = y;
                y = i;
                swap(matrix, i, j, x, y);
            }
        }
    }
    void swap(int[][] matrix, int x1, int y1, int x2, int y2) {
        matrix[x1][y1] ^= matrix[x2][y2];
        matrix[x2][y2] ^= matrix[x1][y1];
        matrix[x1][y1] ^= matrix[x2][y2];
    }

    /* ---------
         字符串
       --------- */

    // 反转字符串
    public void reverseString(char[] s) {
        int len = s.length;
        for (int i = 0; i < len / 2; i++) {
            char temp = s[i];
            s[i] = s[len - 1 - i];
            s[len - 1 - i] = temp;
        }
    }

    // 整数反转
    public int reverse(int x) {
        char[] s = String.valueOf(Math.abs(x)).toCharArray();
        int len = s.length;
        for (int i = 0; i < len / 2; i++) {
            char temp = s[i];
            s[i] = s[len - 1 - i];
            s[len - 1 - i] = temp;
        }
        try {
            int num = Integer.valueOf(String.valueOf(s));
            return x < 0 ? 0 - num: num;
        } catch (Exception e) {
            return 0;
        }
    }

    // 字符串中的第一个唯一字符
    public int firstUniqChar(String s) {
        int index = -1;
        for(char ch = 'a'; ch <= 'z'; ch++) {
            int beginIndex = s.indexOf(ch);
            if (beginIndex != -1 && beginIndex == s.lastIndexOf(ch)) {
                index = index == -1 ? beginIndex : Math.min(beginIndex, index);
            }
        }
        return index;
    }

    // 有效的字母异位词
    public boolean isAnagram(String s, String t) {
        char[] chars1 = s.toCharArray();
        char[] chars2 = t.toCharArray();
        Arrays.sort(chars1);
        Arrays.sort(chars2);
        return String.valueOf(chars1).equals(String.valueOf(chars2));
    }

    // 验证回文字符串
    public boolean isPalindrome(String s) {
        s = s.toLowerCase();
        int i = 0, j = s.length() - 1;
        while (i < j) {
            char ch1 = s.charAt(i);
            if ((ch1 < 'a' || ch1 > 'z') && (ch1 < '0' || ch1 > '9')) {
                i++;
                continue;
            }
            char ch2 = s.charAt(j);
            if ((ch2 < 'a' || ch2 > 'z') && (ch2 < '0' || ch2 > '9')) {
                j--;
                continue;
            }
            if (ch1 != ch2) {
                return false;
            }
            i++;
            j--;
        }
        return true;
    }

    // 字符串转整数
    public int myAtoi(String str) {
        String s = str.trim();
        if (s.length() == 0 || s == null) {
            return 0;
        }
        int index = s.contains(" ") ? s.indexOf(" ") : s.length();
        char c = s.charAt(0);
        boolean flag = true;
        if (c != '-' && c != '+') {
            flag = false;
            if (c < '0' || c > '9') {
                return 0;
            }
        }
        for (int i = (flag ? 1 : 0); i < index; i++) {
            c = s.charAt(i);
            if (c < '0' || c > '9') {
                index = i;
            }
        }
        if (flag && index == 1) {
            return 0;
        }
        try {
            return Integer.parseInt(s.substring(0, index));
        } catch (Exception e) {
            return s.charAt(0) == '-' ? Integer.MIN_VALUE : Integer.MAX_VALUE;
        }
    }

    // 外观数列
    public String countAndSay(int n) {
        if (n == 1) {
            return "1";
        }
        StringBuffer sb = new StringBuffer();
        String s = countAndSay(n - 1);
        int length = s.length();
        int j = 0;
        for (int i = 1; i < length; i++) {
            if (s.charAt(i) != s.charAt(j)) {
                sb.append(i - j).append(s.charAt(j));
                j = i;
            }
        }
        return sb.append(length - j).append(s.charAt(j)).toString();
    }

    //最长公共前缀
    public String longestCommonPrefix(String[] strs) {
        if (strs.length < 1 || strs == null) {
            return "";
        }
        if (strs.length == 1) {
            return strs[0];
        }
        int len = strs.length;
        int strLen = strs[0].length();
        int index = 0;
        try {
            for (; index < strLen; index++) {
                char ch = strs[0].charAt(index);
                for (int j = 1; j < len; j++) {
                    if (strs[j].charAt(index) != ch) {
                        return strs[0].substring(0, index);
                    }
                }
            }
        } catch (Exception e) {
            return strs[0].substring(0, index);
        }
        return strs[0];
    }

    /* ---------
         链表
       --------- */

    //删除链表中的节点
    public void deleteNode(ListNode node) {
        node.val = node.next.val;
        node.next = node.next.next;
    }

    //删除链表的倒数第N个节点
    public ListNode removeNthFromEnd(ListNode head, int n) {
        Map<Integer, ListNode> map = new HashMap<>();
        ListNode node = head;
        int index = 0;
        while (node != null) {
            map.put(index++, node);
            node = node.next;
        }
        if (index - n == 0) {
            return head.next;
        }
        node = map.get(index - n);
        node.next = node.next.next;
        return head;
    }

    //反转链表
    public ListNode reverseList(ListNode head) {
        if (head == null) {
            return null;
        }
        ListNode temp = new ListNode(head.val);
        ListNode ret = temp;
        ListNode node = head.next;
        while (node != null) {
            ret = new ListNode(node.val);
            ret.next = temp;
            temp = ret;
            node = node.next;
        }
        return ret;
    }

    //合并两个有序链表
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        if (l1 == null && l2 == null) {
            return null;
        } else if (l1 == null) {
            return l2;
        } else if (l2 == null) {
            return l1;
        }
        ListNode head = new ListNode(0);
        ListNode node = head;
        while(l1 != null && l2 != null) {
            if (l1.val < l2.val) {
                node.next = new ListNode(l1.val);
                l1 = l1.next;
            } else {
                node.next = new ListNode(l2.val);
                l2 = l2.next;
            }
            node = node.next;
        }
        if (l1 == null) {
            node.next = l2;
        } else if (l2 == null) {
            node.next = l1;
        }
        return head;
    }

    //回文链表
    public boolean isPalindrome(ListNode head) {
        if (head == null || head.next == null) {
            return true;
        }
        if (head.next.next == null) {
            return head.val == head.next.val;
        }
        ListNode slow = head;
        ListNode fast = head;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        slow = reverseList(slow);
        ListNode temp = slow;
        ListNode node = head;
        while (temp != null) {
            if (node.val == temp.val) {
                node = node.next;
                temp = temp.next;
            } else {
                return false;
            }
        }
        reverseList(slow);
        return true;
    }

    //环形链表
    public boolean hasCycle(ListNode head) {
        ListNode fast = head, slow = head;
        while (fast != null && fast.next != null) {
            fast = fast.next.next;
            slow = slow.next;
            if (slow == fast) {
                return true;
            }
        }
        return false;
    }

    /* ---------
         树
       --------- */

    //二叉树的最大深度
    public int maxDepth(TreeNode root) {
        if (root == null) {
            return 0;
        }
        if (root.left == null && root.right == null) {
            return 1;
        } else if (root.left == null) {
            return maxDepth(root.right) + 1;
        } else if (root.right == null) {
            return maxDepth(root.left) + 1;
        } else {
            return Math.max(maxDepth(root.left), maxDepth(root.right)) + 1;
        }
    }

    //验证二叉搜索树
    long pre = Long.MIN_VALUE;
    public boolean isValidBST(TreeNode root) {
        if (root == null) return true;
        if (!isValidBST(root.left)) return false;
        if (root.val <= pre) return false;
        pre = root.val;
        return isValidBST(root.right);
    }

    //对称二叉树
    public boolean isSymmetric(TreeNode root) {
        return check(root, root);
    }

    boolean check(TreeNode p, TreeNode q) {
        if (p == null && q == null) {
            return true;
        }
        if (p == null || q == null) {
            return false;
        }
        return p.val == q.val && check(p.left, q.right) && check(p.right, q.left);
    }

    //二叉树的层序遍历
    public List<List<Integer>> levelOrder(TreeNode root) {
        if (root == null) {
            List<List<Integer>> ret = new LinkedList<>();
            return ret;
        }
        Queue<TreeNode> queue = new LinkedList<>();
        Map<Integer, List<TreeNode>> map = new HashMap<>();
        Map<TreeNode, Integer> depthMap = new HashMap<>();
        queue.add(root);
        List<TreeNode> list = new ArrayList<>();
        list.add(root);
        map.put(1, list);
        depthMap.put(root, 1);
        while (!queue.isEmpty()) {
            TreeNode node = queue.poll();
            int sonDepth = depthMap.get(node) + 1;
            if (!map.containsKey(sonDepth)) {
                list = new ArrayList<>();
                map.put(sonDepth, list);
            }
            list = map.get(sonDepth);
            if (node.left != null) {
                list.add(node.left);
                queue.add(node.left);
                depthMap.put(node.left, sonDepth);
            }
            if (node.right != null) {
                list.add(node.right);
                queue.add(node.right);
                depthMap.put(node.right, sonDepth);
            }
        }
        List<List<Integer>> ret = new LinkedList<>();
        for (int i = 1; i < map.size(); i++) {
            list = map.get(i);
            List<Integer> list1 = new LinkedList<>();
            for (TreeNode treeNode : list) {
                list1.add(treeNode.val);
            }
            ret.add(list1);
        }
        return ret;
    }

    //将有序数组转化为二叉树
    public TreeNode sortedArrayToBST(int[] nums) {
        return sort(nums, 0, nums.length);
    }

    TreeNode sort(int[] nums, int left, int right) {
        if (left >= right) return null;
        int middle = left + (right - left) / 2;
        TreeNode node = new TreeNode(nums[middle]);
        if (left == middle) {
            return node;
        }
        node.left = sort(nums, left, middle);
        node.right = sort(nums, middle + 1, right);
        return node;
    }

    /* ----------
        排序和搜索
       --------- */

    //合并两个有序数组
    public void merge(int[] nums1, int m, int[] nums2, int n) {
        int p = m + n - 1, i = m - 1, j = n - 1;
        while (i >= 0 && j >= 0) {
            if (nums1[i] <= nums2[j]) nums1[p--] = nums2[j--];
            else nums1[p--] = nums1[i--];
        }
        for (int k = j - 1; k >= 0; k--) nums1[p--] = nums1[k];
        for ( ; j >= 0; j--) nums1[j] = nums2[j];
    }

    //第一个错误的版本
    public int firstBadVersion(int n) {
        if (n < 1) return 0;
        if (n == 1) return isBadVersion(n) ? 1 : 0;
        int left = 0, right = n;
        while (left < right) {
            int middle = left + (right - left) / 2;
            if (isBadVersion(middle)) {
                right = middle;
            } else {
                left = middle + 1;
            }
        }
        return left;
    }

    boolean isBadVersion(int version) {
        return true;
    }

    /* ---------
        动态规划
       --------- */

    //爬楼梯
    public int climbStairs(int n) {
        if (n <= 2) {
            return n;
        }
        int[] dp = new int[n + 1];
        dp[0] = 0;
        dp[1] = 1;
        dp[2] = 2;
        for (int i = 3; i < n + 1; i++) {
            dp[i] = dp[i - 1] + dp[i - 2];
        }
        return dp[n];
    }

    //买卖股票的最佳时机
    public int maxProfit1(int[] prices) {
        if (prices.length <= 1) {
            return 0;
        }
        int maxProfit = 0;
        int minValue = prices[0];
        for (int i = 1; i < prices.length; i++) {
            if (prices[i] < minValue) {
                minValue = prices[i];
            } else {
                maxProfit = Math.max(maxProfit, prices[i] - minValue);
            }
        }
        return maxProfit;
    }

    //最大子序和
    public int maxSubArray(int[] nums) {
        if (nums.length == 0) {
            return 0;
        } else if (nums.length == 1) {
            return nums[0];
        }
        int[] dp = new int[nums.length];
        int max = dp[0] = nums[0];
        for (int i = 1; i < dp.length; i++) {
            dp[i] = dp[i - 1] > 0 ? dp[i - 1] + nums[i] : nums[i];
            if (max < dp[i]) {
                max = dp[i];
            }
        }
        return max;
    }

    //打家劫舍
    public int rob(int[] nums) {
        if (nums.length == 0) {
            return 0;
        }
        if (nums.length == 1) {
            return nums[0];
        }
        if (nums.length == 2) {
            return Math.max(nums[0], nums[1]);
        }
        int[] dp = new int[nums.length];
        dp[0] = nums[0];
        dp[1] = Math.max(nums[0], nums[1]);
        for (int i = 2; i < nums.length; i++) {
            dp[i] = Math.max(dp[i - 1], dp[i - 2] + nums[i]);
        }
        return dp[nums.length - 1];
    }

    /* ---------
        设计问题
       --------- */

    //打乱数组
    class ShuffleArray {

        private int[] origin;

        public ShuffleArray(int[] nums) {
            origin = new int[nums.length];
            for (int i = 0; i < nums.length; i++) {
                origin[i] = nums[i];
            }
        }

        public int[] reset() {
            return origin;
        }

        public int[] shuffle() {
            Random random = new Random();
            int[] nums = origin.clone();
            for (int i = nums.length - 1; i >= 0; i--) {
                int index = random.nextInt(i + 1);
                int temp = nums[i] ^ nums[index];
                nums[index] ^= temp;
                nums[i] ^= temp;
            }
            return nums;
        }

    }

    //最小栈
    class MinStack {

        /** initialize your data structure here. */
        class Node {

            int value;
            Node next;

            public Node(int value) {
                this.value = value;
                this.next = null;
            }

            public Node(int value, Node next) {
                this.value = value;
                this.next = next;
            }

        }

        private Node head;

        /** initialize your data structure here. */
        public MinStack() {
            this.head = new Node(0);
        }

        public void push(int x) {
            if (head.next == null) {
                head.next = new Node(x);
            } else {
                Node node = head.next;
                head.next = new Node(x, node);
            }
        }

        public void pop() {
            if (head.next.next == null) {
                head.next = null;
            } else {
                Node node = head.next.next;
                head.next = node;
            }
        }

        public int top() {
            return head.next.value;
        }

        public int getMin() {
            int min = Integer.MAX_VALUE;
            Node node = head.next;
            while (node != null) {
                if (node.value < min) {
                    min = node.value;
                }
                node = node.next;
            }
            return min;
        }
    }

    /* ---------
          数学
       --------- */
    public List<String> fizzBuzz(int n) {
        if (n <= 0) {
            return null;
        }
        List<String> list = new LinkedList<>();
        for (int i = 1; i < n + 1; i++) {
            if (i % 3 == 0 && i % 5 == 0) {
                list.add("FizzBuzz");
            } else if (i % 3 == 0) {
                list.add("Fizz");
            } else if (i % 5 == 0) {
                list.add("Buzz");
            } else {
                list.add(String.valueOf(i));
            }
        }
        return list;
    }

    //计数质数
    public int countPrimes(int n) {
        if (n <= 2) return 0;
        int count = 0;
        boolean flag;
        for (int i = 2; i < n; i++) {
            flag = true;
            int m = (int)Math.sqrt(i);
            for (int j = 2; j < m + 1; j++) {
                if (i % j == 0) {
                    flag = false;
                    break;
                }
            }
            if (flag) {
                count++;
            }
        }
        return count;
    }

    //3的幂次方
    public boolean isPowerOfThree(int n) {
        if (n == 1) return true;
        boolean flag = n > 1;
        while (n > 1) {
            if (n % 3 == 0) {
                n /= 3;
            } else {
                flag = false;
                break;
            }
        }
        return flag;
    }

    //罗马数字转整数
    public int romanToInt(String s) {
        Map<Character, Integer> map = new HashMap<>();
        map.put('I', 1);
        map.put('V', 5);
        map.put('X', 10);
        map.put('L', 50);
        map.put('C', 100);
        map.put('D', 500);
        map.put('M', 1000);
        int ret = 0;
        for (int i = 0; i < s.length() - 1; i++) {
            int num1 = map.get(s.charAt(i));
            int num2 = map.get(s.charAt(i + 1));
            if (num1 < num2) {
                ret -= num1;
            } else {
                ret += num1;
            }
        }
        ret += map.get(s.charAt(s.length() - 1));
        return ret;
    }

    /* ---------
          其他
       --------- */

    //位1的个数
    public int hammingWeight(int n) {
        int ret = 0;
        for(char ch: Integer.toBinaryString(n).toCharArray()) {
            if (ch == '1') {
                ret++;
            }
        }
        return ret;
    }

    //汉明距离
    public int hammingDistance(int x, int y) {
        int n = x ^ y;
        int count = 0;
        while (n != 0) {
            n &= n - 1;
            count++;
        }
        return count;
    }

    //颠倒二进制位
    public int reverseBits(int n) {
        int power = 31, ret = 0;
        while (n != 0) {
            ret += (n & 1) << power;
            n >>>= 1;
            power -= 1;
        }
        return ret;
    }

    //杨辉三角
    public List<List<Integer>> generate(int numRows) {
        List<List<Integer>> ret = new ArrayList<>();
        if (numRows == 0) return ret;
        List<Integer> list = new ArrayList<>();
        list.add(1);
        ret.add(list);
        if (numRows == 1) return ret;
        list = new ArrayList<>();
        list.add(1);
        list.add(1);
        ret.add(list);
        if (numRows == 2) return ret;
        for (int i = 2; i < numRows; i++) {
            list = new ArrayList<>();
            List<Integer> pre = ret.get(i - 1);
            for (int j = 0; j <= i; j++) {
                if (j == 0 || j == i) {
                    list.add(1);
                    continue;
                }
                list.add(pre.get(j - 1) + pre.get(j));
            }
            ret.add(list);
        }
        return ret;
    }

    //有效的括号
    public boolean isValid(String s) {
        if (s == null || s.length() == 0) return true;
        if (s.length() % 2 == 1) return false;
        Deque<Character> stack = new LinkedList<>();
        Map<Character, Character> map = new HashMap<>();
        map.put('}', '{');
        map.put(']', '[');
        map.put(')', '(');
        for (char c : s.toCharArray()) {
            if (c == '(' || c == '[' || c == '{') {
                stack.addFirst(c);
            } else if (c == ')' || c == ']' || c == '}') {
                if (stack.isEmpty()) return false;
                char ch = stack.removeFirst();
                if (ch != map.get(c)) {
                    return false;
                }
            }
        }
        return stack.isEmpty();
    }

    //缺失的数字
    public int missingNumber(int[] nums) {
        int n = nums.length;
        int m = n * (n + 1) / 2;
        for (int num : nums) {
            m -= num;
        }
        return m;
    }

    /* ======================================
                        中级算法
       ====================================== */

    /* ---------
       数组和字符串
       --------- */

}