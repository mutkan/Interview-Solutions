/* The Love-Letter Mystery
    HackerRank
    Coursera 12/24/2018
    */
class Result {

    /*
     * Complete the 'mystery' function below.
     *
     * The function is expected to return an INTEGER_ARRAY.
     * The function accepts STRING_ARRAY letter as parameter.
     */

    public static List<Integer> mystery(List<String> letter) {
    // Write your code here
        List<Integer> result = new ArrayList<>();
        for (int i=0; i<letter.size(); i++) {
            result.add(mysteryHelper(letter.get(i)));
        }
        return result;
    }

    public static int mysteryHelper(String s) {
        int operation = 0;
        for (int i=0, j=s.length()-1; i<s.length()/2; i++, j--) {
            operation += Math.abs(s.charAt(i) - s.charAt(j));
        }
        return operation;
    }
    
}