
/*
* 剑指Offer
* */
public class Offer {

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
