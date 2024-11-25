using Newtonsoft.Json;
using System.Collections.Concurrent;
using System.IO.Compression;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;

namespace TinyGPT.SimpleDecoder.Trainer
{
	internal static class Program
	{
		private const int magicTokenClasses = 3;
		private const int maxContextSize = 4096;
		private const int encbufsize = maxContextSize - 1;
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

			int maxlen = 0;
			int tokenclasses = 0;
			foreach (KeyValuePair<string, ushort> keyValuePair in dict)
			{
				maxlen = Math.Max(maxlen, keyValuePair.Key.Length);
				tokenclasses = Math.Max(keyValuePair.Value, tokenclasses);
			}
			//5 magic token types
			//[START_GPT], [END_GPT], [WIKI_SEPERATOR], [MASK]
			tokenclasses += magicTokenClasses + 1;
			int tokenClasses2 = tokenclasses;
			Console.WriteLine("Optimizing dictionary...");
			IReadOnlyDictionary<string, OptimizedTokenizerEntry>? dict1 = OptimizeDictionary(dict);



			Console.WriteLine("Loading ELI5 + WikiQA question answering dataset...");
			Queue<string> dataqueue = new Queue<string>();
			//File.ReadAllText(datadir + "QuestionAnsweringV2.jsonl.deflate")

			using (StreamReader reader = new StreamReader(new DeflateStream(new FileStream(datadir + "QuestionAnsweringV2.jsonl.deflate", FileMode.Open, FileAccess.Read, FileShare.Read, 16777216, FileOptions.SequentialScan), CompressionMode.Decompress, false), Encoding.UTF8, false, 16777216, false))
			{
			read:
				string? line = reader.ReadLine();
				if (line is { })
				{
					dataqueue.Enqueue(line);
					goto read;
				}
			}
			string[]? questionanswering = dataqueue.ToArray();
			int wqlen2 = questionanswering.Length;

			Console.WriteLine("Loading simple english wikipedia dataset...");
			string[]? wikiarticles = File.ReadAllLines(datadir + "simplewiki-latest.jsonl");


			int wikilen2 = wikiarticles.Length;

			Console.WriteLine("Starting dataset tokenizers...");
			int wqlength = wqlen2;
			int wikilen = wikilen2;

			ConcurrentBag<ushort[]>? alldata = new();
			//ConcurrentBag<int[]>? classcounters = new();
			int threads = Environment.ProcessorCount;
			int wikisplit = 0;
			int loadprogress = 0;
			int wikiloadprogress = 0;
			Thread[] thrlist = new Thread[threads];


			for (int z = 0; z < threads; ++z)
			{
				int az = z;
				Thread thread = new Thread(() =>
				{
					int za = az;
					StringBuilder sb = new StringBuilder("Tokenized ");
					Span<ushort> encbuffer = stackalloc ushort[maxContextSize];
					encbuffer[0] = 0;
					Span<ushort> encbuffer2 = encbuffer[1..];
					int mywqlen = wqlength;
					string str1 = "Tokenized {0}/" + mywqlen + " question-answer pairs";
					int mywikilen = wikilen;
					//int[] counter = new int[tokenClasses2];
					//classcounters.Add(counter);
					//int sa2 = suboptimalSkipInitialTokens + 2;

					while (true)
					{
						int a = Interlocked.Increment(ref loadprogress);
						if (a > mywqlen)
						{
							break;
						}
						a -= 1;
						string raw = questionanswering[a];
						bool suboptimal = raw[0] == '!'; //suboptimal flag
						if (suboptimal)
						{
							raw = raw.Substring(1);
						}
						string[]? pair = JsonConvert.DeserializeObject<string[]>(raw);
						if (pair is null)
						{
							continue;
						}



						int size1 = Tokenize(dict1, encbuffer2, pair[1], maxlen, magicTokenClasses);
						if (size1 == 0)
						{
							continue;
						}
						if (size1 < encbufsize)
						{
							encbuffer2[size1++] = 1; //GPT-to-user context switch
						}



						alldata.Add(encbuffer[..(size1 + 1)].ToArray());


						if ((a & 4095) == 4095)
						{
							Console.WriteLine(str1, a);
						}

					}


					str1 = "Tokenized {0}/" + mywikilen + " simple english wikipedia articles";

					while (true)
					{
						int a = Interlocked.Increment(ref wikiloadprogress);
						if (a > mywikilen)
						{
							return;
						}
						a -= 1;
						WikipediaArticle? wikipediaArticle = JsonConvert.DeserializeObject<WikipediaArticle>(wikiarticles[a]);
						if (wikipediaArticle is null)
						{
							continue;
						}
						string wikititle = wikipediaArticle.title;
						string lowertitle = wikititle.ToLower();

						//skip useless lists (TinyGPT is horrible with dealing with those)
						if (lowertitle.StartsWith("list of"))
						{
							continue;
						}
						if (lowertitle.StartsWith("lists of"))
						{
							continue;
						}

						//Span<ushort> encbuffer3 = encbuffer2[size2..];

						string[] section_texts = wikipediaArticle.section_texts;
						string[] section_titles = wikipediaArticle.section_titles;
						int len = Math.Min(section_texts.Length, section_titles.Length);
						for (int segmentid = 0; segmentid < len; ++segmentid)
						{
							string text = section_texts[segmentid];
							if (text.Length < 64)
							{
								continue; //too short to be useful
							}
							string title = section_titles[segmentid];

							//TinyGPT does not handle citations and references well
							switch (title.ToLower())
							{
								case "see also":
								case "references":
									continue;
							}


							int size1 = Tokenize(dict1, encbuffer2, text, maxlen, magicTokenClasses);
							if (size1 == 0)
							{
								continue;
							}
							if (size1 < encbufsize)
							{
								encbuffer2[size1++] = 1; //GPT-to-user context switch
							}



							alldata.Add(encbuffer[..(size1 + 1)].ToArray());




						}
						if ((a & 4095) == 4095)
						{
							Console.WriteLine(str1, a);
						}
					}
				});
				thread.Name = "Dataset tokenizer thread #" + z;
				thread.IsBackground = true;
				thrlist[z] = thread;
				thread.Start();
			}

			Console.WriteLine("Preparing sparse collocation counter...");

			Dictionary<ushort, CountWrapper>[] keyValuePairs = new Dictionary<ushort, CountWrapper>[tokenclasses];
			uint[] tokenCounts = new uint[tokenclasses];
			for (int i = 0; i < tokenclasses;){
				keyValuePairs[i++] = new();
			}


			Console.WriteLine("Waiting for dataset tokenization to complete...");
			foreach (Thread thr in thrlist)
			{
				thr.Join();
			}

			Console.WriteLine("Counting collocations...");
#pragma warning disable CS8600 // Converting null literal or possible null value to non-nullable type.
			while (alldata.TryTake(out ushort[] arr))
			{

				for (int i = 0, stop = arr.Length - 1; i < stop;)
				{
					ushort a = arr[i];
					ushort b = arr[++i];
					++tokenCounts[a];
					Dictionary<ushort, CountWrapper> keyValuePairs1 = keyValuePairs[a];
					if (keyValuePairs1.TryGetValue(b, out CountWrapper countWrapper))
					{
						++countWrapper.val;
					}
					else
					{
						keyValuePairs1.Add(b, new CountWrapper());
					}
				}
			}
			Console.WriteLine("Creating writestream...");

			BufferedStream bf = new BufferedStream(new DeflateStream(new FileStream(save, FileMode.Append | FileMode.Create, FileAccess.Write, FileShare.None, 65536 * 256, FileOptions.SequentialScan), CompressionLevel.SmallestSize, false), 65536 * 256);

			Span<byte> span = stackalloc byte[10];
			ref ushort upf = ref MemoryMarshal.Cast<byte, ushort>(span)[0];
			ref double upd = ref MemoryMarshal.Cast<byte, double>(span.Slice(2,8))[0];





			Console.WriteLine("Saving sparse collocation distributions...");
			for (int i = 0; i < tokenclasses; ++i)
			{
				uint div1 = tokenCounts[i];
				if (div1 == 0)
				{
					continue;
				}
				double div2 = div1;
				upf = (ushort)i;
				bf.Write(span);


				foreach (KeyValuePair<ushort, CountWrapper> kvp in keyValuePairs[i])
				{
					upf = kvp.Key;
					upd = (kvp.Value.val + 1) / div2;
					bf.Write(span);
				}

				//HACK: the [START_GPT] token is NEVER generated
				//so we can use it to mean "end sparse dict"
				upf = 0;
				bf.Write(span);

			}
			upf = 1;
			bf.Write(span);
			bf.Flush();
			bf.Dispose();




		}
		
	}

}