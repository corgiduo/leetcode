import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

/*
* 程序员面试金典
* */
public class Interview {

    //02.02 返回倒数第k个节点
    public int kthToLast(ListNode head, int k) {
        if (head == null) return 0;
        if (head.next == null) return head.val;
        ListNode node = head;
        for (int i = 0; i < k; i++) {
            head = head.next;
        }
        while (head != null) {
            head = head.next;
            node = node.next;
        }
        return node.val;
    }

    //04.02 最小高度树
    public TreeNode sortedArrayToBST(int[] nums) {
        return sortedArrayToBST(nums, 0, nums.length - 1);
    }

    TreeNode sortedArrayToBST(int[] nums, int start, int end) {
        if (start == end + 1) return null;
        int mid = start + (end - start) / 2;
        TreeNode node = new TreeNode(nums[mid]);
        node.left = sortedArrayToBST(nums, start, mid - 1);
        node.right = sortedArrayToBST(nums, mid + 1, end);
        return node;
    }

    //16.10 生存人数
    public int maxAliveYear(int[] birth, int[] death) {
        int[] alive = new int[102];
        for (int i = 0; i < birth.length; i++) {
            alive[birth[i] - 1900]++;
            alive[death[i] - 1899]--;
        }
        int ret = birth[0], max = 1;
        for (int i = 1; i < alive.length; i++) {
            alive[i] = alive[i - 1] + alive[i];
            if (alive[i] > max) {
                max = alive[i];
                ret = i + 1900;
            }
        }
        return ret;
    }

    //16.19. 水域大小
    private int sum = 0;
    public int[] pondSizes(int[][] land) {
        int row = land.length, col = land[0].length;
        List<Integer> list = new LinkedList<>();
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                if (land[i][j] == 0) {
                    sum = 0;
                    dfs(land, i, j);
                    list.add(sum);
                }
            }
        }
        int[] ret = new int[list.size()];
        int i = 0;
        for (Integer integer : list) {
            ret[i++] = integer;
        }
        Arrays.sort(ret);
        return ret;
    }

    public void dfs(int[][] land, int i, int j) {
        int row = land.length, col = land[0].length;
        if (i < 0 || i >= row || j < 0 || j >= col || land[i][j] != 0) return;
        land[i][j] = -1;
        sum++;
        dfs(land, i - 1, j + 1);
        dfs(land, i - 1, j);
        dfs(land, i - 1, j - 1);
        dfs(land, i, j + 1);
        dfs(land, i, j - 1);
        dfs(land, i + 1, j + 1);
        dfs(land, i + 1, j);
        dfs(land, i + 1, j - 1);
    }

}
