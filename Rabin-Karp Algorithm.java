/* Rabin-Karp Algorithm for Pattern Searching

Rabin Karp algorithm matches the hash value of the pattern with the hash value of current substring of text, and if the hash values match then only it starts matching individual characters. */

public class Mian {
	// d is the number of characters in the input alphabet
	public final static int d = 256;

	// q -> A prime number
	static void search (String pattern, String text, int q) {
		int M = pattern.length();
		int N = text.length();
		int i, j;
		int p = 0; // hash value for pattern
		int t = 0; // hash value for text
		int h = 1;

		for (i = 0; i < M-1; i++) {
			h = (h * d) % q;
		}

		// Calculater the hash value of pattern and first window of text
		for (i = 0; i < M; i++) {
			p = (d*p + pattern.charAt(i)) % q;
			t = (d*t + text.charAt(i)) % q;
		}

		// Slide the pattern over text one by one
		for (i = 0; i <= N - M; i++) {
			// Check hash values of current window of text and pattern. If the hash values match then only check for characters one by one
			if (p == t) {
				for (j = 0; j < M; j++) {
					if (text.charAt(i+j) != pattern.charAt(j))	break;
				}
				if (j == M)	System.out.println("Pattern found at index " + i);
			}
			// Calculate hash value for next window of text: Remove leading digit, add trailing digit
			if (i < N-M) {
				t = (d * (t - text.charAt(i) * h) + text.chatAt(i + M)) % q;
				if (t < 0)	t = t + q;
			}
		}
	}
}