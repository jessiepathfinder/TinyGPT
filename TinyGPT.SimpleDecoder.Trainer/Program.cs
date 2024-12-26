using Newtonsoft.Json;
using System.Collections.Concurrent;
using System.IO.Compression;
using System.Linq.Expressions;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;

namespace TinyGPT.SimpleDecoder.Trainer
{
	internal static class Program
	{
		private const int magicTokenClasses = 3;
		private const int counterCompactingTreshold = 1024;

		private static readonly ConcurrentBag<string> strings = new ConcurrentBag<string>();
		private static readonly ConcurrentBag<(ushort[],int)> tokenizedQueue = new ConcurrentBag<(ushort[],int)>();
		private static readonly ConcurrentBag<ushort[]> recycler = new ConcurrentBag<ushort[]>();
		private static readonly SemaphoreSlim loadSemaphore = new SemaphoreSlim(256);
		private static readonly SemaphoreSlim availableSemaphore = new SemaphoreSlim(0);
		private static readonly SemaphoreSlim availableSemaphore2 = new SemaphoreSlim(0);

		private static void LoadThread(IReadOnlyDictionary<string, OptimizedTokenizerEntry> keyValuePairs, string datadir){
			SemaphoreSlim c1 = loadSemaphore;
			SemaphoreSlim c2 = availableSemaphore;
			SemaphoreSlim c3 = availableSemaphore2;
			int corecount = Environment.ProcessorCount;
			if (corecount > 1) corecount -= 1;
			Thread[] threads = new Thread[corecount];
			for(int i = 0; i < corecount; ){
				Thread thr = new Thread(() => TkzThread(keyValuePairs));
				thr.IsBackground = true;
				int ipp = i + 1;
				thr.Name = "Dataset tokenizer thread #" + ipp;
				threads[i] = thr;
				thr.Start();
				i = ipp;
			}
			ConcurrentBag<string> inp = strings;

			using (StreamReader streamReader = new StreamReader(new DeflateStream(new FileStream(datadir + "QuestionAnsweringV2.jsonl.deflate", FileMode.Open, FileAccess.Read, FileShare.Read, 16777216, FileOptions.SequentialScan), CompressionMode.Decompress, false), Encoding.UTF8, false, -1, false))
			{
				while(true){
					c1.Wait();
					string? line = streamReader.ReadLine();
					if (line is null) break;
					if (line.StartsWith('!')) line = line.Substring(1);

					inp.Add((JsonConvert.DeserializeObject<string[]>(line) ?? throw new Exception())[1]);
					c2.Release();
				}
			}
			using (StreamReader streamReader = new StreamReader(new FileStream(datadir + "simplewiki-latest.jsonl", FileMode.Open, FileAccess.Read, FileShare.Read, 16777216, FileOptions.SequentialScan), Encoding.UTF8, false, -1, false))
			{
				while (true)
				{

					string? line = streamReader.ReadLine();
					if (line is null) break;

					WikipediaArticle wikipediaArticle = JsonConvert.DeserializeObject<WikipediaArticle>(line) ?? throw new Exception();
					string lowertitle = wikipediaArticle.title.ToLower();

					//skip useless lists (TinyGPT is horrible with dealing with those)
					if (lowertitle.StartsWith("list of"))
					{
						continue;
					}
					if (lowertitle.StartsWith("lists of"))
					{
						continue;
					}
					string[] texts = wikipediaArticle.section_texts;
					string[] titles = wikipediaArticle.section_titles;

					int txl = texts.Length;
					for(int i = 0; i < txl; ++i){
						switch (titles[i].ToLower())
						{
							case "see also":
							case "references":
								continue;
						}
						c1.Wait();
						inp.Add(texts[i]);
						c2.Release();
					}
				}
			}
			using (StreamReader streamReader = new StreamReader(new GZipStream(new FileStream(datadir + "enwiki-latest-pages.jsonl.gz", FileMode.Open, FileAccess.Read, FileShare.Read, 16777216, FileOptions.SequentialScan), CompressionMode.Decompress), Encoding.UTF8, false, -1, false))
			{
				while (true)
				{

					string? line = streamReader.ReadLine();
					if (line is null) break;

					WikipediaArticle wikipediaArticle = JsonConvert.DeserializeObject<WikipediaArticle>(line) ?? throw new Exception();
					string lowertitle = wikipediaArticle.title.ToLower();

					//skip useless lists (TinyGPT is horrible with dealing with those)
					if (lowertitle.StartsWith("list of"))
					{
						continue;
					}
					if (lowertitle.StartsWith("lists of"))
					{
						continue;
					}
					string[] texts = wikipediaArticle.section_texts;
					string[] titles = wikipediaArticle.section_titles;

					int txl = texts.Length;
					for (int i = 0; i < txl; ++i)
					{
						switch (titles[i].ToLower())
						{
							case "see also":
							case "references":
								continue;
						}
						c1.Wait();
						inp.Add(texts[i]);
						c2.Release();
					}
				}
			}
			for (int i = 0; i < corecount; ++i) c2.Release();
			for (int i = 0; i < corecount;) threads[i++].Join();
			c3.Release();
		}
		private static void TkzThread(IReadOnlyDictionary<string, OptimizedTokenizerEntry> keyValuePairs){
			SemaphoreSlim c1 = loadSemaphore;
			SemaphoreSlim c2 = availableSemaphore;
			SemaphoreSlim c3 = availableSemaphore2;
			ConcurrentBag<string> inp = strings;
			ConcurrentBag<(ushort[],int)> otp = tokenizedQueue;
			

			while (true){
				
				c2.Wait();
				ConcurrentBag<ushort[]> poolbuffer = Program.recycler;
				if (!poolbuffer.TryTake(out ushort[]? tkzbuffer))
				{
					tkzbuffer = new ushort[65536];
				}
				if (tkzbuffer is null) throw new Exception();
				
#pragma warning disable CS8600 // Converting null literal or possible null value to non-nullable type.
				if (!inp.TryTake(out string str)) return;
#pragma warning restore CS8600 // Converting null literal or possible null value to non-nullable type.
				int i = Tokenize(keyValuePairs, tkzbuffer, str, 128, magicTokenClasses);
				if (i == 0){
					c1.Release();
					poolbuffer.Add(tkzbuffer);
					continue;
				}
				if(i < 65536){
					tkzbuffer[i++] = 1;
				}
				otp.Add((tkzbuffer, i));

				

				c3.Release();
			}
		}



		[JsonObject(MemberSerialization.Fields)]
		private sealed class WikipediaArticle
		{
			//SUPPRESS WARNINGS since fields will be reflectively set
#pragma warning disable CS8618 // Non-nullable field must contain a non-null value when exiting constructor. Consider declaring as nullable.
#pragma warning disable CS0649
			public string title;
			public string[] section_titles;
			public string[] section_texts;
		}
		private sealed class CountWrapper{
			public uint val;
		}
		public static Dictionary<string, OptimizedTokenizerEntry> OptimizeDictionary(IReadOnlyDictionary<string, ushort> input)
		{
			string[] keys = input.Keys.ToArray();
			int len = keys.Length;
			Dictionary<string, OptimizedTokenizerEntry> thedict = new Dictionary<string, OptimizedTokenizerEntry>(len);

			foreach (KeyValuePair<string, ushort> kvp in input)
			{
				bool fastret = true;
				string str = kvp.Key;

				for (int i = 0, sl = str.Length; i < len;)
				{
					string str2 = keys[i++];
					if (str2.Length > sl && str2.StartsWith(str))
					{
						fastret = false;
						break;
					}
				}
				thedict.Add(str, new OptimizedTokenizerEntry(kvp.Value, fastret));
			}
			return thedict;
		}
		public readonly struct OptimizedTokenizerEntry
		{
			public readonly ushort value;
			public readonly bool fastret;

			public OptimizedTokenizerEntry(ushort value, bool fastret)
			{
				this.value = value;
				this.fastret = fastret;
			}
		}

		public static int Tokenize(IReadOnlyDictionary<string, OptimizedTokenizerEntry> dict, Span<ushort> output, ReadOnlySpan<char> str, int maxtokensize, int specialTokenClasses)
		{
			if (maxtokensize < 1)
			{
				throw new ArgumentOutOfRangeException(nameof(maxtokensize));
			}
			int pos = 0;
			int ctr2 = 0;
			for (int len = str.Length, outlen = output.Length; ctr2 < len & pos < outlen;)
			{
				StringBuilder sb = new StringBuilder();
				int token = -1;
				for (int i = ctr2++, stop = Math.Min(i + maxtokensize, len); i < stop; ++i)
				{
					sb.Append(str[i]);
					if (dict.TryGetValue(sb.ToString(), out OptimizedTokenizerEntry val))
					{
						token = val.value;
						ctr2 = i + 1;
						if (val.fastret)
						{
							break;
						}
					}
				}
				if (token > -1)
				{
					output[pos++] = (ushort)(token + specialTokenClasses);
				}
			}
			return pos;
		}
		private static void Main(string[] args)
		{
			string datadir = args[0];
			string save = args[1];
			if (!datadir.EndsWith(Path.DirectorySeparatorChar))
			{
				datadir += Path.DirectorySeparatorChar;
			}
			Console.WriteLine("Loading dictionary...");
			IReadOnlyDictionary<string, ushort>? dict = JsonConvert.DeserializeObject<IReadOnlyDictionary<string, ushort>>(File.ReadAllText(datadir + "encoder.json"));
			if (dict is null)
			{
				Console.WriteLine("Null encoder dictionary");
				return;
			}

			int tokenclasses = 0;
			foreach (KeyValuePair<string, ushort> keyValuePair in dict)
			{
				tokenclasses = Math.Max(keyValuePair.Value, tokenclasses);
			}
			//3 magic token types
			//[START_GPT], [END_GPT], [WIKI_SEPERATOR]
			tokenclasses += magicTokenClasses + 1;
			Console.WriteLine("Optimizing dictionary...");
			IReadOnlyDictionary<string, OptimizedTokenizerEntry> dict1 = OptimizeDictionary(dict);

			Console.WriteLine("Launching background workers...");
			Thread thr = new Thread(() => LoadThread(dict1, datadir));
			thr.IsBackground = true;
			thr.Name = "Dataset IO thread";
			thr.Start();

			Console.WriteLine("Preparing to start training...");
			SemaphoreSlim c1 = loadSemaphore;
			SemaphoreSlim c2 = availableSemaphore;
			SemaphoreSlim c3 = availableSemaphore2;
			ConcurrentBag<(ushort[], int)> otp = tokenizedQueue;
			ConcurrentBag<ushort[]> recycler = Program.recycler;
			Span<ushort> arrfix = stackalloc ushort[counterCompactingTreshold];
			ulong[] tokenCounts = new ulong[tokenclasses];
			Dictionary<ushort, CountWrapper>[] keyValuePairs = new Dictionary<ushort, CountWrapper>[tokenclasses];
			for (int i = 0; i < tokenclasses;) keyValuePairs[i++] = new();
			uint[]?[] efficentCounters = new uint[]?[tokenclasses];
			ushort[]?[] efficentIndexes = new ushort[]?[tokenclasses];
			for (int i = 0; i < tokenclasses;) keyValuePairs[i++] = new();
			ulong pct = 0;

			Console.WriteLine("Counting collocations...");
#pragma warning disable CS8600 // Converting null literal or possible null value to non-nullable type.
			while (true)
			{
				c1.Release();
				c3.Wait();
				if (!otp.TryTake(out (ushort[] arr, int truesize) x1)) break;

				ushort a = 0;
				for (int i = 0; i < x1.truesize;)
				{
					ushort b = x1.arr[i++];
					if (b == 0) throw new Exception("b = 0");
					if (a == 1) throw new Exception("a = 1");
					++tokenCounts[a];
					ushort[]? myindexes = efficentIndexes[a];
					if(myindexes is { }){
						int ind2 = myindexes.AsSpan().BinarySearch(b);
						if (ind2 > -1){
							++(efficentCounters[a] ?? throw new Exception())[ind2];
							goto skip;
						}
					}

					Dictionary<ushort, CountWrapper> keyValuePairs1 = keyValuePairs[a];

					if (keyValuePairs1.TryGetValue(b, out CountWrapper countWrapper))
					{
						++countWrapper.val;
					}
					else
					{

						keyValuePairs1.Add(b, new CountWrapper());
						if(keyValuePairs1.Count == counterCompactingTreshold){
							if(myindexes is null){
								ushort[] clonedind = keyValuePairs1.Keys.ToArray();
								clonedind.AsSpan().Sort();
								uint[] uints = new uint[counterCompactingTreshold];
								for(int x = 0; x < counterCompactingTreshold; ++x){
									uints[x] = keyValuePairs1[clonedind[x]].val;
								}
								efficentIndexes[a] = clonedind;
								efficentCounters[a] = uints;
							} else{
								
								
								
								int ctr = 0;
								foreach(ushort key in keyValuePairs1.Keys){
									arrfix[ctr++] = key;
								}
								arrfix.Sort();
								
								
								int oldlen = myindexes.Length;
								int newlen = oldlen + counterCompactingTreshold;
								ushort[] newindexes = new ushort[newlen];
								uint[] newctrs = new uint[newlen];
								uint[] uints = efficentCounters[a] ?? throw new Exception();

								for(int ctr1 = 0,ai = 0, bi = 0; ctr1 < newlen; ){
									ushort myind;
									uint myvalue;
									if(ai == counterCompactingTreshold){
										myind = myindexes[bi];
										goto sortright;
									}
									if(bi == oldlen){
										myind = arrfix[ai++];
										goto sortleft;
									}
									ushort lind = arrfix[ai];
									ushort rind = myindexes[bi];
									if(lind > rind){
										myind = rind;
										goto sortright;
									}
									++ai;
									myind = lind;
								sortleft:
									myvalue = keyValuePairs1[myind].val;
									goto donesort;
								sortright:
									myvalue = uints[bi++];
								donesort:
									newctrs[ctr1] = myvalue;
									newindexes[ctr1++] = myind;
									
								}

								efficentIndexes[a] = newindexes;
								efficentCounters[a] = newctrs;
							}

							keyValuePairs1.Clear();
						}
					}
				skip:
					a = b;
				}
				recycler.Add(x1.arr);

				if (pct % 1000 == 0) Console.WriteLine("Processed {0} articles", pct);
				++pct;
			}
			Console.WriteLine("Creating writestream...");

			BufferedStream bf = new BufferedStream(new DeflateStream(new FileStream(save, FileMode.Append | FileMode.Create, FileAccess.Write, FileShare.None, 65536 * 256, FileOptions.SequentialScan), CompressionLevel.SmallestSize, false), 65536 * 256);

			Span<byte> span = stackalloc byte[10];
			ref ushort upf = ref MemoryMarshal.Cast<byte, ushort>(span)[0];
			ref double upd = ref MemoryMarshal.Cast<byte, double>(span.Slice(2,8))[0];





			Console.WriteLine("Saving sparse collocation distributions...");
			for (int i = 0; i < tokenclasses; ++i)
			{
				ulong div1 = tokenCounts[i];
				if (div1 == 0)
				{
					continue;
				}
				double div2 = div1;
				upf = (ushort)i;
				bf.Write(span);

				Dictionary<ushort, CountWrapper> mydict = keyValuePairs[i];
				foreach (KeyValuePair<ushort, CountWrapper> kvp in mydict)
				{
					upf = kvp.Key;
					upd = (kvp.Value.val + 1) / div2;
					bf.Write(span);
				}

				ushort[]? myindexes = efficentIndexes[i];
				if(myindexes is { }){
					uint[] myvalues = efficentCounters[i] ?? throw new Exception();
					for(int c = 0, stop = myindexes.Length; c < stop; ++c){
						upf = myindexes[c];
						upd = (myvalues[c] + 1) / div2;
						bf.Write(span);
					}
				}
				

				//HACK: the [START_GPT] token is NEVER generated
				//so we can use it to mean "end sparse dict"
				upf = 0;
				bf.Write(span);

			}
			bf.Flush();
			bf.Dispose();




		}
		
	}

}