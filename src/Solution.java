import ds.ListNode;
import ds.Node;
import ds.TreeNode;

import java.util.*;

public class Solution {

    //5. 最长回文子串
    public String longestPalindrome(String s) {
        if (s == null) return null;
        if (s.length() <= 1) return s;
        int l = 0, r = 0, max = 1;
        int len = s.length();
        boolean[][] dp = new boolean[len][len];
        dp[0][0] = true;
        for (int i = 1; i < len; i++) {
            dp[i][i] = true;
            dp[i][i - 1] = true;
        }
        for (int i = len - 1; i >= 0; i--) {
            for (int j = 0; j < i; j++) {
                int right = len - i + j;
                if (s.charAt(j) == s.charAt(right)) {
                    if (dp[j + 1][right - 1]) {
                        dp[j][right] = true;
                        if (right - j + 1 > max) {
                            max = right - j + 1;
                            l = j;
                            r = right;
                        }
                    }
                }
            }
        }
        return s.substring(l, r + 1);
    }

    //62. 不同路径
    public int uniquePaths(int m, int n) {
        int[][] dp = new int[m][n];
        for (int i = 0; i < m; i++) {
            dp[i][0] = 1;
        }
        for (int i = 0; i < n; i++) {
            dp[0][i] = 1;
        }
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
            }
        }
        return dp[m - 1][n - 1];
    }

    //63 不同路径II
    public int uniquePathsWithObstacles(int[][] obstacleGrid) {
        int m = obstacleGrid.length;
        int n = obstacleGrid[0].length;
        int[][] dp = new int[m][n];
        boolean flag = false;
        for (int i = 0; i < m; i++) {
            if (flag) {
                dp[i][0] = 0;
                continue;
            }
            if (obstacleGrid[i][0] == 1) {
                dp[i][0] = 0;
                flag = true;
                continue;
            }
            dp[i][0] = 1;
        }
        flag = false;
        for (int i = 0; i < n; i++) {
            if (flag) {
                dp[0][i] = 0;
                continue;
            }
            if (obstacleGrid[0][i] == 1) {
                dp[0][i] = 0;
                flag = true;
                continue;
            }
            dp[0][i] = 1;
        }
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                if (obstacleGrid[i][j] == 1) {
                    dp[i][j] = 0;
                    continue;
                }
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
            }
        }
        return dp[m - 1][n - 1];
    }

    //64. 最小路径和
    public int minPathSum(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        int[][] dp = new int[m][n];
        dp[0][0] = grid[0][0];
        for (int i = 1; i < m; i++) {
            dp[i][0] = grid[i][0] + dp[i - 1][0];
        }
        for (int i = 1; i < n; i++) {
            dp[0][i] = grid[0][i] + dp[0][i - 1];
        }
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                dp[i][j] = Math.min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j];
            }
        }
        return dp[m - 1][n - 1];
    }

    //91. 解码方法
    public int numDecodings(String s) {
        if (s == null || s.length() == 0) return 0;
        if (s.charAt(0) == '0') return 0;
        if (s.length() == 1) return 1;
        for (int i = 0; i < s.length() - 1; i++) {
            if (s.charAt(i) == '0' && s.charAt(i + 1) == '0') {
                return 0;
            }
            if (s.charAt(i + 1) == '0' && Integer.parseInt(s.substring(i, i + 2)) > 26) {
                return 0;
            }
        }
        int[] dp = new int[s.length()];
        dp[0] = 1;
        dp[1] = 1;
        if (Integer.parseInt(s.substring(0, 2)) <= 26 && s.charAt(1) != '0') {
            dp[1] = 2;
        }
        for (int i = 2; i < s.length(); i++) {
            int n = Integer.parseInt(s.substring(i - 1, i + 1));
            if (s.charAt(i) == '0' && n <= 26) {
                dp[i] = dp[i - 2];
            } else if (s.charAt(i) == '0' && n > 26) {
                return 0;
            } else {
                if (s.charAt(i - 1) == '0') {
                    dp[i] = dp[i - 1];
                } else if (n <= 26) {
                    dp[i] = dp[i - 1] + dp[i - 2];
                } else {
                    dp[i] = dp[i - 1];
                }
            }
        }
        return dp[s.length() - 1];
    }

    //152. 乘积最大子数组
    public int maxProduct(int[] nums) {
        if (nums.length == 0) return 0;
        if (nums.length == 1) return nums[0];
        int[] maxDp = new int[nums.length];
        int[] minDp = new int[nums.length];
        maxDp[0] = nums[0];
        minDp[0] = nums[0];
        int max = maxDp[0];
        for (int i = 1; i < nums.length; i++) {
            if (nums[i] < 0) {
                maxDp[i] = Math.max(nums[i], minDp[i - 1] * nums[i]);
                minDp[i] = Math.min(nums[i], maxDp[i - 1] * nums[i]);
            } else {
                maxDp[i] = Math.max(nums[i], maxDp[i - 1] * nums[i]);
                minDp[i] = Math.min(nums[i], minDp[i - 1] * nums[i]);
            }
            max = Math.max(maxDp[i], max);
        }
        return max;
    }

    //213. 打家劫舍II
    public int rob(int[] nums) {
        if (nums.length == 0) {
            return 0;
        } else if (nums.length == 1) {
            return nums[0];
        } else if (nums.length == 2) {
            return Math.max(nums[0], nums[1]);
        } else if (nums.length == 3) {
            int max = Math.max(nums[0], nums[1]);
            max = Math.max(max, nums[2]);
            return max;
        } else if (nums.length == 4) {
            return Math.max(nums[0] + nums[2], nums[1] + nums[3]);
        }
        int len = nums.length;
        int[] dp1 = new int[len];
        int[] dp2 = new int[len];
        dp1[2] = nums[2];
        dp1[3] = Math.max(nums[2], nums[3]);
        dp2[1] = nums[1];
        dp2[2] = Math.max(nums[1], nums[2]);
        int max1 = Math.max(dp1[2], dp1[3]), max2 = Math.max(dp2[1], dp2[2]);
        for (int i = 4; i < len - 1; i++) {
            dp1[i] = Math.max(dp1[i - 2] + nums[i], dp1[i - 1]);
            max1 = Math.max(dp1[i], max1);
        }
        max1 += nums[0];
        for (int i = 3; i < len; i++) {
            dp2[i] = Math.max(dp2[i - 2] + nums[i], dp2[i - 1]);
            max2 = Math.max(dp2[i], max2);
        }
        return Math.max(max1, max2);
    }

    //114. 二叉树展开为链表
    public void flatten(TreeNode root) {
        if (root == null) return;
        Deque<TreeNode> stack = new LinkedList<>();
        stack.addFirst(root);
        TreeNode preNode = null, curNode;
        while (!stack.isEmpty()) {
            curNode = stack.pollFirst();
            if (preNode != null) {
                preNode.right = curNode;
            }
            if (curNode.right != null) {
                stack.addFirst(curNode.right);
            }
            if (curNode.left != null) {
                stack.addFirst(curNode.left);
            }
            curNode.left = null;
            preNode = curNode;
        }
    }

    //415. 字符串相加
    public String addStrings(String num1, String num2) {
        int i = num1.length() - 1, j = num2.length() - 1, cur = 0;
        StringBuilder sb = new StringBuilder();
        while (i >= 0 || j >= 0 || cur > 0) {
            int x = i < 0 ? 0 : num1.charAt(i--) - '0';
            int y = j < 0 ? 0 : num2.charAt(j--) - '0';
            sb.append((x + y + cur) % 10);
            cur = (x + y + cur) / 10;
        }
        return sb.reverse().toString();
    }

    //120. 三角形最小路径和
    public int minimumTotal(List<List<Integer>> triangle) {
        int n = triangle.size();
        int[] f = new int[n];
        f[0] = triangle.get(0).get(0);
        for (int i = 1; i < n; ++i) {
            f[i] = f[i - 1] + triangle.get(i).get(i);
            for (int j = i - 1; j > 0; --j) {
                f[j] = Math.min(f[j - 1], f[j]) + triangle.get(i).get(j);
            }
            f[0] += triangle.get(i).get(0);
        }
        int minTotal = f[0];
        for (int i = 1; i < n; ++i) {
            minTotal = Math.min(minTotal, f[i]);
        }
        return minTotal;
    }

    //96. 不同的二叉搜索树
    public int numTrees(int n) {
        if (n <= 2) {
            return n;
        }
        int[] dp = new int[n + 1];
        dp[0] = dp[1] = 1;
        dp[2] = 2;
        for (int i = 3; i < n + 1; i++) {
            dp[i] = dp[i - 1] * 2;
            for (int j = 1; j < i - 1; j++) {
                dp[i] += dp[i - 1 - j] * dp[j];
            }
        }
        return dp[n];
    }

    //95. 不同的二叉搜索树II
    public List<TreeNode> generateTrees(int n) {
        if (n == 0) return new LinkedList<>();
        return generateTrees(1, n);
    }

    List<TreeNode> generateTrees(int start, int end) {
        List<TreeNode> list = new LinkedList<>();
        if (start > end) {
            list.add(null);
            return list;
        }
        for (int i = start; i <= end; i++) {
            List<TreeNode> left = generateTrees(start, i - 1);
            List<TreeNode> right = generateTrees(i + 1, end);
            for (TreeNode l : left) {
                for (TreeNode r : right) {
                    TreeNode node = new TreeNode(i);
                    node.left = l;
                    node.right = r;
                    list.add(node);
                }
            }
        }
        return list;
    }

    //221. 最大正方形
    public int maximalSquare(char[][] matrix) {
        if (matrix.length == 0 || matrix[0].length == 0) return 0;
        if (matrix.length == 1) {
            for (int i = 0; i < matrix[0].length; i++) {
                if (matrix[0][i] == '1') {
                    return 1;
                }
            }
            return 0;
        }
        if (matrix[0].length == 1) {
            for (char[] chars : matrix) {
                if (chars[0] == '1') {
                    return 1;
                }
            }
            return 0;
        }
        int row = matrix.length, col = matrix[0].length;
        int[][] dp = new int[row][col];
        int max = 0;
        for (int i = 0; i < col; i++) {
            if (matrix[0][i] == '1') {
                max = 1;
            }
            dp[0][i] = matrix[0][i] - '0';
        }
        for (int i = 0; i < row; i++) {
            if (matrix[i][0] == '1') max = 1;
            dp[i][0] = matrix[i][0] - '0';
        }
        for (int i = 1; i < row; i++) {
            for (int j = 1; j < col; j++) {
                if (matrix[i][j] == '1') {
                    dp[i][j] = Math.min(dp[i - 1][j], dp[i][j - 1]);
                    dp[i][j] = Math.min(dp[i][j], dp[i - 1][j - 1]);
                    dp[i][j]++;
                    max = Math.max(max, dp[i][j]);
                }
            }
        }
        return max * max;
    }

    //696. 计数二进制子串
    public int countBinarySubstrings(String s) {
        if (s == null || s.length() <= 1) return 0;
        int len = 0, i = 1;
        int nums[] = new int[s.length()];
        nums[len] = 1;
        while (i < s.length()) {
            if (s.charAt(i - 1) != s.charAt(i)) {
                nums[++len] = 1;
            } else {
                nums[len]++;
            }
            i++;
        }
        int ret = 0;
        for (i = 0; i < len; i++) {
            ret += Math.min(nums[i], nums[i + 1]);
        }
        return ret;
    }

    //130. 被围绕的区域
    public void solve(char[][] board) {
        if (board.length == 0 || board[0].length == 0) return;
        int row = board.length, col = board[0].length;
        boolean[][] flags = new boolean[row][col];
        for (int i = 0; i < col; i++) {
            dfs(board, 0, i);
            dfs(board, row - 1, i);
        }
        for (int i = 0; i < row; i++) {
            dfs(board, i, 0);
            dfs(board, i, col - 1);
        }
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                if (board[i][j] == 'A') board[i][j] = 'O';
                else if (board[i][j] == 'O') board[i][j] = 'X';
            }
        }
    }

    void dfs(char[][] board, int x, int y) {
        int row = board.length, col = board[0].length;
        if (x < 0 || x >= row || y < 0 || y >= col || board[x][y] != 'O') return;
        board[x][y] = 'A';
        dfs(board, x, y + 1);
        dfs(board, x, y - 1);
        dfs(board, x - 1, y);
        dfs(board, x + 1, y);
    }

    //133. 克隆图
    public Node cloneGraph(Node node) {
        if (node == null) return node;
        Node head = new Node(node.val);
        if (node.neighbors.isEmpty()) return head;
        Deque<Node> stack = new LinkedList<>();
        Map<Node, Node> map = new HashMap<>();
        stack.addFirst(node);
        map.put(node, head);
        while (!stack.isEmpty()) {
            node = stack.removeFirst();
            for (Node neighbor : node.neighbors) {
                if (!map.containsKey(neighbor)) {
                    stack.addFirst(neighbor);
                    Node temp = new Node(neighbor.val);
                    map.put(neighbor, temp);
                }
            }
        }
        for (Map.Entry<Node, Node> entry : map.entrySet()) {
            Node clone = entry.getValue();
            Node origin = entry.getKey();
            for (Node neighbor : origin.neighbors) {
                clone.neighbors.add(map.get(neighbor));
            }
        }
        return head;
    }

    //43. 字符串相乘
    public String multiply(String num1, String num2) {
        if (num1.equals("0") || num2.equals("0")) {
            return "0";
        }
        int m = num1.length(), n = num2.length();
        int[] ansArr = new int[m + n];
        for (int i = m - 1; i >= 0; i--) {
            int x = num1.charAt(i) - '0';
            for (int j = n - 1; j >= 0; j--) {
                int y = num2.charAt(j) - '0';
                ansArr[i + j + 1] += x * y;
            }
        }
        for (int i = m + n - 1; i > 0; i--) {
            ansArr[i - 1] += ansArr[i] / 10;
            ansArr[i] %= 10;
        }
        int index = ansArr[0] == 0 ? 1 : 0;
        StringBuffer ans = new StringBuffer();
        while (index < m + n) {
            ans.append(ansArr[index]);
            index++;
        }
        return ans.toString();
    }

    //20. 有效括号
    public boolean isValid(String s) {
        if (s == null || s.length() == 0) return true;
        if (s.length() % 2 == 1) return false;
        Deque<Character> stack = new LinkedList<>();
        Map<Character, Character> map = new HashMap<>();
        map.put('(', ')');
        map.put('[', ']');
        map.put('{', '}');
        char ch = s.charAt(0);
        if (ch != '(' && ch != '[' && ch != '{') return false;
        stack.addFirst(ch);
        for (int i = 1; i < s.length(); i++) {
            ch = s.charAt(i);
            if (ch == '(' || ch == '[' || ch == '{') {
                stack.addFirst(ch);
            } else if (ch != map.get(stack.removeFirst())) {
                return false;
            }
        }
        return stack.isEmpty();
    }

    //733. 图像渲染
    public int[][] floodFill(int[][] image, int sr, int sc, int newColor) {
        if (image[sr][sc] == newColor) return image;
        floodFill(image, sr, sc, image[sr][sc], newColor);
        return image;
    }

    public void floodFill(int[][] image, int sr, int sc, int oldColor, int newColor) {
        if (sr < 0 || sr >= image.length || sc < 0 || sc >= image[0].length || image[sr][sc] != oldColor) return;
        image[sr][sc] = newColor;
        floodFill(image, sr - 1, sc, oldColor, newColor);
        floodFill(image, sr + 1, sc, oldColor, newColor);
        floodFill(image, sr, sc - 1, oldColor, newColor);
        floodFill(image, sr, sc + 1, oldColor, newColor);
    }

    //973. 最接近原点的K个点
    public int[][] kClosest(int[][] points, int K) {
        int[] dists = new int[points.length];
        for (int i = 0; i < points.length; i++) {
            dists[i] = points[i][0] * points[i][0] + points[i][1] * points[i][1];
        }
        Arrays.sort(dists);
        int n = dists[K - 1];
        int[][] ret = new int[K][2];
        int j = 0;
        for (int i = 0; i < points.length; i++) {
            int dist = points[i][0] * points[i][0] + points[i][1] * points[i][1];
            if (dist <= n) {
                ret[j++] = points[i];
            }
        }
        return ret;
    }

    //3. 无重复字符的最长子串
    public int lengthOfLongestSubstring(String s) {
        if (s == null || s.length() == 0) return 0;
        int ret = 1;
        int[] dp = new int[s.length()];
        Map<Character, Integer> map = new HashMap<>();
        dp[0] = 1;
        map.put(s.charAt(0), 0);
        for (int i = 1; i < s.length(); i++) {
            char ch = s.charAt(i);
            dp[i] = Math.min(dp[i - 1] + 1, i - map.getOrDefault(ch, -1));
            ret = Math.max(dp[i], ret);
            map.put(ch, i);
        }
        return ret;
    }

    //1190. 反转每对括号间的子串
    public String reverseParentheses(String s) {
        if (s == null || s.length() == 0) return s;
        char[] str = s.toCharArray();
        Deque<Integer> stack = new LinkedList<>();
        for (int i = 0; i < str.length; i++) {
            if (str[i] == '(') {
                stack.addFirst(i);
            } else if (str[i] == ')') {
                int lastIndex = stack.removeFirst();
                reverse(str, lastIndex, i + 1);
            }
        }
        return String.valueOf(str).replace("(", "").replace(")", "");
    }

    public void reverse(char[] str, int start, int end) {
        int middle = (end - start) / 2;
        for (int i = 0; i < middle; i++) {
            char temp = str[start + i];
            str[start + i] = str[end - i - 1];
            str[end - i - 1] = temp;
        }
    }

    //459. 重复的子字符串
    public boolean repeatedSubstringPattern(String s) {
        String str = s + s;
        return str.substring(1,     str.length() - 1).contains(s);
    }

    //406. 根据身高重建队列
    public int[][] reconstructQueue(int[][] people) {
        Arrays.sort(people, (o1, o2) -> (o1[0] == o2[0] ? o1[1] - o2[1] : o2[0] - o1[0]));
        List<int[]> list = new LinkedList<>();
        for (int[] i: people) {
            list.add(i[1], i);
        }
        return list.toArray(new int[list.size()][]);
//        Arrays.sort(people, (o1, o2) -> {
//            return o1[0] == o2[0] ? o2[1] - o1[1] : o1[0] - o2[0];
//        });
//        int[][] ret = new int[people.length][2];
//        boolean[] flag = new boolean[people.length];
//        for (int i = 0; i < people.length; i++) {
//            int index = people[i][1];
//            int k = 0, j = 0;
//            for (j = 0; j < people.length; j++) {
//                if (k == index && !flag[j]) {
//                    break;
//                }
//                if (!flag[j]) {
//                    k++;
//                }
//            }
//            flag[j] = true;
//            ret[j] = people[i];
//        }
//        return ret;
    }

    //110. 平衡二叉树
    public boolean isBalanced(TreeNode root) {
        if (root == null || (root.left == null && root.right == null)) return true;
        return Math.abs(height(root.left) - height(root.right)) <= 1
                && isBalanced(root.left) && isBalanced(root.right);
    }

    public int height(TreeNode root) {
        if (root == null) return 0;
        return Math.max(height(root.left), height(root.right)) + 1;
    }

    //300. 最长上升子序列
    public int lengthOfLIS(int[] nums) {
        if (nums.length == 0) return 0;
        int[] dp = new int[nums.length];
        dp[0] = 1;
        int max = 1;
        for (int i = 1; i < nums.length; i++) {
            for (int j = 0; j < i; j++) {
                if (nums[i] > nums[j]) {
                    dp[i] = Math.max(dp[i], dp[j]);
                }
            }
            dp[i]++;
            max = Math.max(dp[i], max);
        }
        return max;
    }

    //520. 检测大写字母
    public boolean detectCapitalUse(String word) {
        if (word == null || word.length() <= 1) return true;
        boolean flag = word.charAt(0) >= 'A' && word.charAt(0) <= 'Z';
        if (!flag) {
            for (int i = 1; i < word.length(); i++) {
                if (word.charAt(i) < 'a' || word.charAt(i) > 'z')
                    return false;
            }
        } else {
            flag = word.charAt(1) >= 'A' && word.charAt(1) <= 'Z';
            for (int i = 0; i < word.length(); i++) {
                if (flag) {
                    if (word.charAt(i) < 'A' || word.charAt(i) > 'Z')
                        return false;
                } else {
                    if (word.charAt(i) < 'a' || word.charAt(i) > 'z')
                        return false;
                }
            }
        }
        return true;
    }

    //599. 两个列表的最小索引总和
    public String[] findRestaurant(String[] list1, String[] list2) {
        Map<String, Integer> map = new HashMap<>();
        for (int i = 0; i < list1.length; i++) {
            map.put(list1[i], i);
        }
        int minIndex = Integer.MAX_VALUE;
        List<String> ret = new LinkedList<>();
        for (int i = 0; i < list2.length; i++) {
            String str = list2[i];
            if (map.containsKey(str)) {
                int index = map.get(str) + i;
                if (index < minIndex) {
                    minIndex = index;
                    ret.clear();
                    ret.add(str);
                } else if (index == minIndex) {
                    ret.add(str);
                }
            }
        }
        return ret.toArray(new String[0]);
    }

    //495. 提莫攻击
    public int findPoisonedDuration(int[] timeSeries, int duration) {
        if (timeSeries.length == 0) return 0;
        if (timeSeries.length == 1) return duration;
        int start =  timeSeries[0], sum = duration;
        for (int i = 1; i < timeSeries.length; i++) {
            if (timeSeries[i] < start + duration) {
                sum += timeSeries[i] - start;
            } else {
                sum += duration;
            }
            start = timeSeries[i];
        }
        return sum;
    }

    //111. 二叉树的最小深度
    public int minDepth(TreeNode root) {
        if (root == null) return 0;
        if (root.left == null && root.right == null) return 1;
        if (root.left == null) return minDepth(root.right) + 1;
        if (root.right == null) return minDepth(root.left) + 1;
        return Math.min(minDepth(root.left), minDepth(root.right)) + 1;
    }

    //1038. 从二叉搜索树到更大和树
    int sum = 0;
    public TreeNode bstToGst(TreeNode root) {
        if (root != null) {
            bstToGst(root.right);
            sum += root.val;
            root.val = sum;
            bstToGst(root.left);
        }
        return root;
    }

    //1399. 统计最大组的数目
    public int countLargestGroup(int n) {
        if (n < 10) return n;
        Map<Integer, List<Integer>> map = new HashMap<>();
        int ret = 1, count = 1;
        for (int i = 1; i <= n; i++) {
            int sum = sum(i);
            if (!map.containsKey(sum)) {
                List<Integer> list = new LinkedList<>();
                list.add(i);
                map.put(sum, list);
            } else {
                map.get(sum).add(i);
                if (map.get(sum).size() > ret) {
                    ret = map.get(sum).size();
                    count = 1;
                } else if (map.get(sum).size() == ret) {
                    count++;
                }
            }
        }
        return count;
    }

    int sum(int n) {
        int sum = 0;
        while (n > 0) {
            sum += n % 10;
            n /= 10;
        }
        return sum;
    }

    //109. 有序链表转换二叉搜索树
    public TreeNode sortedListToBST(ListNode head) {
        if (head == null) return null;
        List<Integer> list = new LinkedList<>();
        while (head != null) {
            list.add(head.val);
            head = head.next;
        }
        int[] nums = new int[list.size()];
        int i = 0;
        for (Integer integer : list) {
            nums[i++] = integer;
        }
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

    //647. 回文子串
    public int countSubstrings(String s) {
        int n = s.length(), ans = 0;
        for (int i = 0; i < 2 * n - 1; ++i) {
            int l = i / 2, r = i / 2 + i % 2;
            while (l >= 0 && r < n && s.charAt(l) == s.charAt(r)) {
                --l;
                ++r;
                ++ans;
            }
        }
        return ans;
    }

    //529. 扫雷游戏
    public char[][] updateBoard(char[][] board, int[] click) {
        int row = board.length, col = board[0].length, x = click[0], y = click[1];
        if (x < 0 || x >= row || y < 0 || y >= col) return board;
        if (board[x][y] == 'M') {
            board[x][y] = 'X';
            return board;
        }
        if (board[x][y] == 'E') {
            int count = 0;
            if (x - 1 >= 0) {
                if (y - 1 >= 0 && board[x - 1][y - 1] == 'M') count++;
                if (board[x - 1][y] == 'M') count++;
                if (y + 1 < col && board[x - 1][y + 1] == 'M') count++;
            }
            if (y - 1 >= 0 && board[x][y - 1] == 'M') count++;
            if (y + 1 < col && board[x][y + 1] == 'M') count++;
            if (x + 1 < row) {
                if (y - 1 >= 0 && board[x + 1][y - 1] == 'M') count++;
                if (board[x + 1][y] == 'M') count++;
                if (y + 1 < col && board[x + 1][y + 1] == 'M') count++;
            }
            if (count != 0) {
                board[x][y] = (char)('0' + count);
                return board;
            }
            board[x][y] = 'B';
            board = updateBoard(board, new int[]{x - 1, y - 1});
            board = updateBoard(board, new int[]{x - 1, y});
            board = updateBoard(board, new int[]{x - 1, y + 1});
            board = updateBoard(board, new int[]{x, y - 1});
            board = updateBoard(board, new int[]{x, y + 1});
            board = updateBoard(board, new int[]{x + 1, y - 1});
            board = updateBoard(board, new int[]{x + 1, y});
            board = updateBoard(board, new int[]{x + 1, y + 1});
        }
        return board;
    }

    //160. 相交链表
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        if (headA == null || headB == null) return null;
        ListNode tailA = headA;
        while (tailA.next != null) {
            tailA = tailA.next;
        }
        tailA.next = headB;
        ListNode fast = headA, slow = headA;
        while (fast != null && fast.next != null) {
            fast = fast.next.next;
            slow = slow.next;
            if (fast == slow) {
                break;
            }
        }
        if (fast != slow) {
            tailA.next = null;
            return null;
        }
        ListNode temp = headA;
        while (temp != slow) {
            temp = temp.next;
            slow = slow.next;
        }
        tailA.next = null;
        return slow;
    }

    //201. 数字范围按位与
    public int rangeBitwiseAnd(int m, int n) {
        while (m < n) n &= (n - 1);
        return n;
    }

    //767. 重构字符串
    public String reorganizeString(String S) {
        int len = S.length();
        char[] ret = new char[len];
        int maxCnt = len / 2 + len % 2; //字符允许出现的最大次数
        int[] cnt = new int[26];
        int maxIndex = 0;   //出现最多次数字符的下标
        for (int i = 0; i < len; i++) {
            int index = S.charAt(i) - 'a';  //当前字符对应下标
            cnt[index]++;
            if (cnt[index] > maxCnt) {  //超出允许次数，无法重构
                return "";
            }
            if (cnt[index] > cnt[maxIndex]) {
                maxIndex = index;
            }
        }
        int index = 0;
        while (cnt[maxIndex]-- > 0) {
            ret[index] = (char)(maxIndex + 'a');
            index += 2;
        }
        for (int i = 0; i < 26; i++) {
            while (cnt[i]-- > 0) {
                if (index >= len) {
                    index = 1;
                }
                ret[index] = (char)(index + 'a');
                index += 2;
            }
        }
        return String.valueOf(ret);
    }

    //34. 在排序数组中查找元素的第一个和最后一个位置
    public int[] searchRange(int[] nums, int target) {
        int len = nums.length;
        if (len == 0 || (len == 1 && target != nums[0]) || (target < nums[0] || target > nums[len - 1])) {
            return new int[]{-1, -1};
        }
        int l = 0, r = len - 1, m = (l + r) / 2;
        while (target != nums[m] && l < r) {
            if (target < nums[m]) {
                r = m - 1;
                m = (l + r) / 2;
            } else {
                l = m + 1;
                m = (l + r) / 2;
            }
        }
        if (target != nums[m]) {
            return new int[]{-1, -1};
        }
        l = m;
        r = m;
        while (l >= 0 && nums[l] == target) {
            l--;
        }
        while (r < len && nums[r] == target) {
            r++;
        }
        int[] ret = new int[2];
        ret[0] = l + 1;
        ret[1] = r - 1;
        return ret;
    }


    //204. 计数质数
    public int countPrimes(int n) {
        int[] isPrime = new int[n];
        Arrays.fill(isPrime, 1);
        int ans = 0;
        for (int i = 2; i < n; ++i) {
            if (isPrime[i] == 1) {
                ans += 1;
                if ((long) i * i < n) {
                    for (int j = i * i; j < n; j += i) {
                        isPrime[j] = 0;
                    }
                }
            }
        }
        return ans;
    }

    //135. 分发糖果
    public int candy(int[] ratings) {
        int len = ratings.length;
        int[] left = new int[len];
        for (int i = 0; i < len; i++) {
            if (i > 0 && ratings[i] > ratings[i - 1]) {
                left[i] = left[i - 1] + 1;
            } else {
                left[i] = 1;
            }
        }
        int right = 0, sum = 0;
        for (int i = len - 1; i >= 0; i--) {
            if (i < len - 1 && ratings[i] > ratings[i + 1]) {
                right++;
            } else {
                right = 1;
            }
            sum += Math.max(left[i], right);
        }
        return sum;
    }

    //85. 最大矩形
    public int maximalRectangle(char[][] matrix) {
        if (matrix.length == 0 || matrix[0].length == 0) {
            return 0;
        }
        int x = matrix.length;
        int y = matrix[0].length;
        int[][] left = new int[x][y];
        int ret = 0;
        for (int i = 0; i < x; i++) {
            for (int j = 0; j < y; j++) {
                if (matrix[i][j] == '1') {
                    if (j == 0) {
                        left[i][j] = 1;
                    } else {
                        left[i][j] = left[i][j - 1] + 1;
                    }
                }
            }
        }
        for (int i = 0; i < x; i++) {
            for (int j = 0; j < y; j++) {
                if (matrix[i][j] == '1') {
                    int width = left[i][j], height = 1;
                    int max = width * height;
                    while (i >= height) {
                        if (left[i - height][j] == 0) {
                            break;
                        }
                        width = Math.min(width, left[i - height][j]);
                        height++;
                        max = Math.max(max, width * height);
                    }
                    ret = Math.max(ret, max);
                }
            }
        }
        return ret;
    }

    //455. 分发饼干
    public int findContentChildren(int[] g, int[] s) {
        int[] g1 = (int[])g.clone(), s1 = (int[])s.clone();
        Arrays.sort(g1);
        Arrays.sort(s1);
        int len1 = g1.length, len2 = s1.length;
        int i = 0, j = 0;
        int count = 0;
        while (i < len1 && j < len2) {
            if (s1[j] >= g1[i]) {
                count++;
                i++;
            }
            j++;
        }
        return count;
    }

    //188. 买卖股票的最佳时机Ⅳ
    public int maxProfit4(int k, int[] prices) {
        int days = prices.length;
        if (days == 0) return 0;
        k = Math.min(k, days / 2);
        int[][] buy = new int[days][k + 1];
        int[][] sell = new int[days][k + 1];
        buy[0][0] = -prices[0];
        sell[0][0] = 0;
        for (int i = 1; i <= k; i++) {
            buy[0][i] = Integer.MIN_VALUE / 2;
            sell[0][i] = Integer.MIN_VALUE / 2;
        }
        for (int i = 1; i < days; i++) {
            buy[i][0] = Math.max(buy[i - 1][0], sell[i - 1][0] - prices[i]);
            for (int j = 1; j <= k; j++) {
                buy[i][j] = Math.max(buy[i - 1][j], sell[i - 1][j] - prices[i]);
                sell[i][j] = Math.max(sell[i - 1][j], buy[i - 1][j - 1] + prices[i]);
            }
        }
        return Arrays.stream(sell[days - 1]).max().getAsInt();
    }

    //2. 两数相加
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        if (l1 == null || l2 == null) {
            return l1 == null ? l2 : l1;
        }
        ListNode n1 = l1, n2 = l2;
        ListNode ret = new ListNode(0);
        ListNode node = ret;
        int flag = 0;
        while (n1 != null || n2 != null) {
            int val = flag;
            if (n1 != null) {
                val += n1.val;
                n1 = n1.next;
            }
            if (n2 != null) {
                val += n2.val;
                n2 = n2.next;
            }
            flag = val / 10;
            val %= 10;
            node.next = new ListNode(val);
            node = node.next;
        }
        if (flag == 1) {
            node.next = new ListNode(1);
        }
        return ret.next;
    }

    //4. 寻找两个正序数组的中位数
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int len1 = nums1.length, len2 = nums2.length;
        int totalLen = len1 + len2;
        int mid = totalLen / 2;
        if (totalLen % 2 == 1) {
            return getKthElement(nums1, nums2, mid + 1);
        } else {
            return (getKthElement(nums1, nums2, mid) + getKthElement(nums1, nums2, mid + 1)) / 2.0;
        }
    }

    int getKthElement(int[] nums1, int[] nums2, int k) {
        int len1 = nums1.length, len2 = nums2.length;
        int i = 0, j = 0;
        int ret = 0;
        while (true) {
            if (i == len1) {
                return nums2[j + k - 1];
            }
            if (j == len2) {
                return nums1[i + k - 1];
            }
            if (k == 1) {
                return Math.min(nums1[i], nums2[j]);
            }
            int half = k / 2;
            int index1 = Math.min(i + half, len1) - 1;
            int index2 = Math.min(j + half, len2) - 1;
            if (nums1[index1] < nums2[index2]) {
                k -= index1 - i + 1;
                i = index1 + 1;
            } else {
                k -= index2 - j + 1;
                j = index2 + 1;
            }
        }
    }

    //509. 斐波那契数
    public int fib(int n) {
        if (n == 0) return 0;
        if (n == 1) return 1;
        int[] arr = new int[n + 1];
        arr[0] = 0;
        arr[1] = 1;
        for (int i = 2; i < n + 1; i++) {
            arr[i] = arr[i - 1] + arr[i - 2];
        }
        return arr[n];
    }

    //830. 较大分组的位置
    public List<List<Integer>> largeGroupPositions(String s) {
        List<List<Integer>> lists = new LinkedList<>();
        if (s.length() < 3) return lists;
        int start = 0, end = 1;
        char cur = s.charAt(0);
        for (int i = 1; i < s.length(); i++) {
            if (s.charAt(i) != cur) {
                end = i - 1;
                if (end - start > 1) {
                    List<Integer> list = new LinkedList();
                    list.add(start);
                    list.add(end);
                    lists.add(list);
                }
                start = i;
                cur = s.charAt(i);
            }
        }
        if (s.charAt(s.length() - 1) == cur) {
            end = s.length() - 1;
            if (end - start > 1) {
                List<Integer> list = new LinkedList();
                list.add(start);
                list.add(end);
                lists.add(list);
            }
        }
        return lists;
    }

    //330. 按要求补齐数组
    public int minPatches(int[] nums, int n) {
        int cnt = 0;
        long x = 1;
        int len = nums.length, index = 0;
        while (x <= n) {
            if (index < len && nums[index] <= x) {
                x += nums[index];
                index++;
            } else {
                x *= 2;
                cnt++;
            }
        }
        return cnt;
    }

    //1046. 最后一块石头的重量
    public int lastStoneWeight(int[] stones) {
        PriorityQueue<Integer> priorityQueue = new PriorityQueue<>((a, b) -> b - a);
        for (int stone : stones) {
            priorityQueue.offer(stone);
        }
        while (priorityQueue.size() > 1) {
            int a = priorityQueue.poll();
            int b = priorityQueue.poll();
            if (a > b) {
                priorityQueue.offer(a - b);
            }
        }
        return priorityQueue.isEmpty() ? 0 : priorityQueue.poll();
    }

    //783. 二叉搜索树节点最小距离
    public int minDiffInBST(TreeNode root) {
        List<Integer> list = new LinkedList<>();
        minDiffInBSTBSF(root, list);
        int min = Integer.MAX_VALUE;
        for (int i = 1; i < list.size(); i++) {
            min = Math.min(min, list.get(i) - list.get(i - 1));
        }
        return min;
    }

    public void minDiffInBSTBSF(TreeNode root, List<Integer> list) {
        if (root == null) {
            return;
        }
        minDiffInBSTBSF(root.left, list);
        list.add(root.val);
        minDiffInBSTBSF(root.right, list);
    }

    //6. Z字形变换
    public String convert(String s, int numRows) {
        int len = s.length();
        if (s.length() == 1 || numRows == 1) return s;
        char[][] chars = new char[numRows][len / 2 + 1];
        int row = 0, col = 0;
        boolean flag = false;
        for (int i = 0; i < len; i++) {
            chars[row][col] = s.charAt(i);
            if (row == numRows - 1 || row == 0) {
                flag = !flag;
            }
            if (flag) {
                row++;
            } else {
                row--;
                col++;
            }
        }
        StringBuffer sb = new StringBuffer();
        for (int i = 0; i < chars.length; i++) {
            for (int j = 0; j < chars[0].length; j++) {
                if (chars[i][j] != '\0') {
                    sb.append(chars[i][j]);
                }
            }
        }
        return sb.toString();
    }

}
