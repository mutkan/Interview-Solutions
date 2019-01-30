// Check if a string is a substring of another

class substringCheck {
	// check if string a is a substring of string b
	static int isSubstring(String a, String b) {
		int m = a.length();
		int n = b.length();

		for (int i = 0; i < n - m; i++) {
			int j;
			for (j = 0; i < m; j++) {
				if (b.charAt(i+j) != a.charAt(j)) {
					break;
				}
			}
			if (j == m) {
				return i;
			}
		}
		return -1;
	}
}