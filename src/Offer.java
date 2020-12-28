import java.util.Arrays;

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
