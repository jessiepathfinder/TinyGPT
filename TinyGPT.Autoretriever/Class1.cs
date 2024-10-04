using System.Text;


//Autoretriever: retrieval-assisted rare words handling
//Will be enabled in a future version.
namespace TinyGPT.Autoretriever
{
	public static class Util
	{
		public static (string,bool) ExpandSee(string str, IReadOnlyDictionary<string, string> dict){
			ReadOnlySpan<char> ros = str.AsSpan();
			StringBuilder? sb = null;
		start:
			int ind = ros.IndexOf("See ");
			if(ind < 1){
				return sb is null ? (str,false) : (sb.ToString(),true);
			}
			if(sb is null) sb = new StringBuilder(Math.Min(4096, str.Length * 2));
			sb.Append(ros[..ind]);
			ReadOnlySpan<char> ros1 = ros[(ind + 3)..];
			int lit = ros1.IndexOf('.');
			if( lit > 0 ){
#pragma warning disable CS8600 // Converting null literal or possible null value to non-nullable type.
				if (dict.TryGetValue(new string(ros1[..lit]).ToLower(), out string str1)){
#pragma warning restore CS8600 // Converting null literal or possible null value to non-nullable type.
					
					sb.Append(str1);
				} else{
					sb.Append(ros.Slice(ind,lit + 1));
				}
				ind += lit;



			}
			ros = ros[(ind + 4)..];
			
			goto start;
		}
		public static Dictionary<string,string> FixDict(IReadOnlyDictionary<string,string> dict){
			Dictionary<string, string> dict1 = new(dict.Count);
			foreach(KeyValuePair<string, string> kvp in dict){
				string key = kvp.Key;
				if (key.Length < 2) continue;
				if (key.StartsWith('-') || key.EndsWith("-")) continue;
				dict1.Add(" " + key.ToLower(), kvp.Value);
			}

			//Remove EXTREMELY COMMON WORDS
			dict1.Remove(" a", out _);
			dict1.Remove(" an", out _);
			dict1.Remove(" is", out _);
			dict1.Remove(" as", out _);
			dict1.Remove(" they", out _);
			dict1.Remove(" their", out _);
			dict1.Remove(" she", out _);
			dict1.Remove(" her", out _);
			dict1.Remove(" him", out _);
			dict1.Remove(" his", out _);
			dict1.Remove(" he", out _);
			dict1.Remove(" not", out _);
			dict1.Remove(" and", out _);
			dict1.Remove(" or", out _);
			dict1.Remove(" the", out _);
			dict1.Remove(" it", out _);
			dict1.Remove(" it's", out _);
			dict1.Remove(" but", out _);
			dict1.Remove(" may", out _);
			dict1.Remove(" because", out _);
			dict1.Remove(" since", out _);
			dict1.Remove(" when", out _);
			dict1.Remove(" never", out _);
			dict1.Remove(" to", out _);
			dict1.Remove(" be", out _);
			dict1.Remove(" then", out _);
			dict1.Remove(" what", out _);
			dict1.Remove(" if", out _);
			dict1.Remove(" could", out _);
			dict1.Remove(" must", out _);
			return dict1;
		}
		public static string ResolveWordMeanings(IReadOnlyDictionary<string,string> dict,string input){

			int len = input.Length;
			if (len == 0) return input;

			string input1 = " " + input.ToLower();
			
			if(char.IsUpper(input[0])) input = char.ToLower(input[0]) + input.Substring(1);
			
			string?[] keys = new string?[len];
			string[] values = new string[len];

			foreach (KeyValuePair<string, string> kvp in dict)
			{
				string key = kvp.Key;
				int fio = input1.IndexOf(key);
				if (fio > -1){
					string? mystrin = keys[fio];
					if(mystrin is { }){
						if (mystrin.Length > key.Length) continue;
					}
					keys[fio] = key;
					values[fio] = kvp.Value;
				}
			}

			StringBuilder stringBuilder = new StringBuilder();
			for(int i = 0; i < len; ++i){
				string? str = keys[i];
				if (str is null) continue;

				
				stringBuilder.Append('#');
				bool exp = true;
				string str2 = str;
				while (exp) (str2, exp) = ExpandSee(str2, dict);
				stringBuilder.AppendLine(str);
				stringBuilder.AppendLine(values[i]);
				stringBuilder.AppendLine();
			}
			stringBuilder.Append("User prompt: ");
			stringBuilder.Append(input);
			return stringBuilder.ToString();
		}
	}
	public sealed class StringOccouranceComparer : IComparer<(string str, string val, int occ)>
	{
		private StringOccouranceComparer() { }
		public static readonly StringOccouranceComparer instance = new StringOccouranceComparer();
		public int Compare((string str, string val, int occ) x, (string str, string val, int occ) y)
		{
			int oc = x.occ.CompareTo(y.occ);
			if (oc != 0) return oc;
			return x.str.Length.CompareTo(y.str.Length);
		}
	}

}