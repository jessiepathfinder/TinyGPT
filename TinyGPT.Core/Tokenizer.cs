using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TinyGPT
{
	public static class Tokenizer
	{
		public static IReadOnlyDictionary<string, float> WordListToDictionary(IEnumerable<string> wordlist)
		{
			Dictionary<string, float> keyValuePairs = new Dictionary<string, float>();

			int i = 65537;
			foreach (string word in wordlist)
			{
				keyValuePairs.Add(word, ++i);
			}
			return keyValuePairs;
		}
		public static int Tokenize(ReadOnlySpan<char> text, int maxtokensizehint, Span<float> tokens, IReadOnlyDictionary<string, float> dict)
		{
			int pos = 0;
			int len = text.Length;
			int maxtokens = tokens.Length;
			if (maxtokens == 0)
			{
				return 0;
			}
			bool capitalize = true;
			bool insertspace = false;
			int ctr = 0;
			while (pos < len & ctr < maxtokens)
			{
				char fc = text[pos++];
				if (insertspace)
				{
					if (fc != ' ')
					{
						//null padding (cancels space)
						tokens[ctr++] = 0;
					}
					insertspace = false;
					continue;
				}
				float token = fc + 2.0f;
				if (capitalize)
				{
					capitalize = false;
					if (char.IsUpper(fc))
					{
						fc = char.ToLower(fc);
					}
					else
					{
						//null padding (cancels capital letter)
						tokens[ctr++] = 0;
						continue;
					}
				}
				else if ((fc == '.' & (pos + 1) < len) && (text[pos] == ' ') && char.IsUpper(text[pos + 1]))
				{
					tokens[ctr++] = 1; //control code for sentence stop
					pos += 2;
					insertspace = true;
					capitalize = true;
					continue;
				}
				StringBuilder stringBuilder = new StringBuilder(fc);
				int bestjmp = pos;
				for (int i = pos, stop = Math.Min(len, pos + maxtokensizehint - 1); i < stop; ++i)
				{
					stringBuilder.Append(text[i]);
					if (dict.TryGetValue(stringBuilder.ToString(), out float tmp))
					{
						token = tmp;
						bestjmp = i;
						insertspace = true;
					}
				}
				tokens[ctr++] = token;
				pos = bestjmp;
			}
			return ctr;
		}
	}

}
