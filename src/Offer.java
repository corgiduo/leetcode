import ds.ListNode;
import ds.TreeNode;

import java.util.Arrays;
import java.util.Deque;
import java.util.LinkedList;

/*
* 剑指Offer
* */
public class Offer {

    //03. 数组中重复的数字
    public int findRepeatNumber(int[] nums) {
        int[] arr = new int[nums.length];
        Arrays.fill(arr, -1);
        int val = -1;
        for (int i = 0; i < nums.length; i++) {
            val = nums[i];
            if (arr[val] == -1) {
                arr[val] = val;
            } else {
                break;
            }
        }
        return val;
    }

    //04. 二维数组中的查找
    public boolean findNumberIn2DArray(int[][] matrix, int target) {
        if (matrix.length == 0 || matrix[0].length == 0) {
            return false;
        }
        int x = 0, y = matrix[0].length - 1;
        while (x < matrix.length && y >= 0) {
            if (target == matrix[x][y]) {
                return true;
            } else if (target > matrix[x][y]) {
                x++;
            } else {
                y--;
            }
        }
        return false;
    }

    //05. 替换空格
    public String replaceSpace(String s) {
        StringBuffer sb = new StringBuffer();
        for (char c : s.toCharArray()) {
            if (c == ' ') {
                sb.append("%20");
            } else {
                sb.append(c);
            }
        }
        return sb.toString();
    }

    //06. 从尾到头打印链表
    public int[] reversePrint(ListNode head) {
        if (head == null) {
            return new int[]{};
        }
        int len = 0;
        Deque<Integer> stack = new LinkedList<>();
        while (head.next != null) {
            stack.offerFirst(head.val);
            head = head.next;
            len++;
        }
        int[] ret = new int[len];
        for (int i = 0; i < len; i++) {
            ret[i] = stack.pollFirst();
        }
        return ret;
    }

    //07. 重建二叉树
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        if(preorder.length == 0 && inorder.length == 0) {
            return null;
        }
        TreeNode root = new TreeNode(preorder[0]);
        int left = 0;
        while (inorder[left] != preorder[0]) {
            left++;
        }
        int[] lp = Arrays.copyOfRange(preorder, 0, left);
        int[] li = Arrays.copyOfRange(inorder, 1, left + 1);
        int[] rp = Arrays.copyOfRange(preorder, left + 1, preorder.length);
        int[] ri = Arrays.copyOfRange(inorder, left + 1, preorder.length);
        root.left = buildTree(lp, li);
        root.right = buildTree(rp, ri);
        return root;
    }

    //09. 用两个栈实现队列
    public class CQueue {

        Deque<Integer> stack1 = new LinkedList<>();
        Deque<Integer> stack2 = new LinkedList<>();

        public CQueue() {

        }

        public void appendTail(int value) {
            stack1.offerFirst(value);
        }

        public int deleteHead() {
            if (stack2.isEmpty()) {
                while (!stack1.isEmpty()) {
                    stack2.offerFirst(stack1.pollFirst());
                }
            }
            return stack2.isEmpty() ? -1 : stack2.pollFirst();
        }

    }

    //10-1. 斐波那契数列
    public int fib(int n) {
        if (n == 0) return 0;
        if (n == 1) return 1;
        int[] nums = new int[n + 1];
        nums[0] = 0;
        nums[1] = 1;
        for (int i = 2; i <= n; i++) {
            nums[i] = (nums[i - 2] + nums[i - 1]) % 1000000007;
        }
        return nums[n];
    }

    //10-2. 青蛙跳台阶问题
    public int numWays(int n) {
        if (n == 0) return 1;
        if (n == 1) return 1;
        int[] nums = new int[n + 1];
        nums[0] = 1;
        nums[1] = 1;
        for (int i = 2; i <= n; i++) {
            nums[i] = (nums[i - 2] + nums[i - 1]) % 1000000007;
        }
        return nums[n];
    }

    //11. 旋转数组的最小数字
    public int minArray(int[] numbers) {
        if (numbers[0] < numbers[numbers.length - 1]) return numbers[0];
        for (int i = 1; i < numbers.length; i++) {
            if (numbers[i] < numbers[i - 1]) return numbers[i];
        }
        return 0;
    }

    //68. 二叉树的最近公共祖先
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null || root == p || root == q) return root;
        TreeNode left = lowestCommonAncestor(root.left, p, q);
        TreeNode right = lowestCommonAncestor(root.right, p, q);
        if (left == null) return right;
        if (right == null) return left;
        return root;
    }

}
