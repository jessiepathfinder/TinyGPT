﻿using Newtonsoft.Json;
using System.Collections.Concurrent;
using System.IO.Compression;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using TinyGPT.Core;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.distributions.constraints;
using static TorchSharp.torch.optim;
using static TorchSharp.torch.optim.lr_scheduler;
using Adam = TinyGPT.Core.Adam;
using Transformer = TinyGPT.Core.Transformer;

namespace TinyGPT.SimpleAttention.Trainer
{
	internal static class Program
	{


		//hyperparameters
		private const int latentTokenSize = 2048;
		private const int maxContextSize = 1025;
		private const int trainingBatches = 2048;
		private const int targetUnlabeledTokensPerBatch = 65536;
		private const int targetLabeledTokensPerBatch = 65536;
		private const int attentionHeads = 16;
		private const int firstTierAttentionDepth = 5;
		private const int magicTokenClasses = 3;
		private const int minimumInputTokens = 2;
		//private const int maxOutputBlockSize = 128;

		private const int supplementalWordEmbeddingsSize = 512;
		//private const int startUnsupervisedTreshold = 512;
		//private const double costScalingTerm = 1024.0;

		[JsonObject(MemberSerialization.Fields)]
		private sealed class WikipediaArticle
		{
			//SUPPRESS WARNINGS since fields will be reflectively set
#pragma warning disable CS8618 // Non-nullable field must contain a non-null value when exiting constructor. Consider declaring as nullable.
#pragma warning disable CS0649
			public string title;
			public string[] section_titles;
			public string[] section_texts;
#pragma warning restore CS0649
#pragma warning restore CS8618 // Non-nullable field must contain a non-null value when exiting constructor. Consider declaring as nullable.
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
			IReadOnlyDictionary<string, OptimizedTokenizerEntry>? dict1 = Misc.OptimizeDictionary(dict);



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

			ConcurrentQueue<ushort[]>? alldata = new();
			//ConcurrentBag<int[]>? classcounters = new();
			int threads = Environment.ProcessorCount;
			Barrier? barrier3 = new Barrier(threads);
			int wikisplit = 0;
			int loadprogress = 0;
			int wikiloadprogress = 0;
			Thread[] thrlist = new Thread[threads];
			int[] emasks = new int[tokenclasses];

			for (int z = 0; z < threads; ++z)
			{
				int az = z;
				Thread thread = new Thread(() =>
				{
					int za = az;
					StringBuilder sb = new StringBuilder("Tokenized ");
					Span<ushort> encbuffer = stackalloc ushort[maxContextSize + 1];
					Span<ushort> encbuffer2 = encbuffer.Slice(1, maxContextSize);
					int mywqlen = wqlength;
					string str1 = "Tokenized {0}/" + mywqlen + " question-answer pairs";
					int mywikilen = wikilen;
					//int[] counter = new int[tokenClasses2];
					//classcounters.Add(counter);
					//int sa2 = suboptimalSkipInitialTokens + 2;
					Dictionary<ushort,bool> keyValuePairs = new Dictionary<ushort,bool>();

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


						int size1 = Transformer.Tokenize(dict1, encbuffer2, pair[0], maxlen, magicTokenClasses);
						if (size1 == maxContextSize)
						{
							continue;
						}
						if (size1 < minimumInputTokens)
						{
							continue;
						}



						encbuffer2[size1++] = 0; //user-to-GPT context switch
						if (size1 == maxContextSize)
						{
							continue;
						}
						int encsize2 = size1;

						//int encsize2 = size1;
						int ctd = Transformer.Tokenize(dict1, encbuffer2[size1..], pair[1], maxlen, magicTokenClasses);
						if (ctd == 0)
						{
							continue;
						}
						size1 += ctd;
						if (size1 < maxContextSize)
						{
							encbuffer2[size1++] = 1; //GPT-to-user context switch
						}
						for (int k = 0; ;)
						{
							ushort us = encbuffer2[k++];
							if (us == 0) break;
							keyValuePairs.TryAdd(us, false);
						}
						encbuffer[0] = (ushort)(encsize2);


						alldata.Enqueue(encbuffer[..(size1 + 1)].ToArray());


						if ((a & 4095) == 4095)
						{
							Console.WriteLine(str1, a);
						}

					}

					barrier3.SignalAndWait();
					if (za == 0)
					{
						wikisplit = alldata.Count;
					}
					barrier3.SignalAndWait();
					if (za == 0)
					{
						barrier3.Dispose();
						barrier3 = null;
					}
					str1 = "Tokenized {0}/" + mywikilen + " simple english wikipedia articles";

					while (true)
					{
						int a = Interlocked.Increment(ref wikiloadprogress);
						if (a > mywikilen)
						{
							break;
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
						int size2 = Transformer.Tokenize(dict1, encbuffer2, wikititle, maxlen, magicTokenClasses);
						if (size2 == maxContextSize)
						{
							continue;
						}
						if (size2 == 0)
						{
							continue;
						}

						encbuffer2[size2++] = 2; //wikipedia article retrieval task

						if (size2 == maxContextSize)
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
							int size1 = Transformer.Tokenize(dict1, encbuffer2[size2..], title, maxlen, magicTokenClasses);
							if (size1 == 0)
							{
								continue;
							}
							size1 += size2;
							if (size1 == maxContextSize)
							{
								continue;
							}

							encbuffer2[size1++] = 0; //[START_GPT]
							if (size1 == maxContextSize)
							{
								continue;
							}

							int ctd = Transformer.Tokenize(dict1, encbuffer2[size1..], text.Replace("'''", null).Replace("''", null), maxlen, magicTokenClasses);
							if (ctd == 0)
							{
								continue;
							}
							encbuffer[0] = (ushort)(size1);
							size1 += ctd;
							if (size1 < maxContextSize)
							{
								encbuffer2[size1++] = 1; //GPT-to-user context switch
							}



							for(int k = 0; ; ){
								ushort us = encbuffer2[k++];
								if (us == 0) break;
								keyValuePairs.TryAdd(us,false);
							}
							alldata.Enqueue(encbuffer[..(size1 + 1)].ToArray());



						}
						if ((a & 4095) == 4095)
						{
							Console.WriteLine(str1, a);
						}
					}
					int[] arry2 = emasks;
					foreach (ushort s in keyValuePairs.Keys) Interlocked.Exchange(ref arry2[s], 1);
				});
				thread.Name = "Dataset tokenizer thread #" + z;
				thread.IsBackground = true;
				thrlist[z] = thread;
				thread.Start();
			}



			Console.WriteLine("Initializing model...");
			InitializeDeviceType(DeviceType.CUDA);
			backends.cuda.matmul.allow_tf32 = true;
			backends.cuda.matmul.allow_fp16_reduced_precision_reduction = false;
			backends.cuda.enable_math_sdp(false);
			backends.cuda.enable_flash_sdp(true);
			backends.cudnn.allow_tf32 = true;

			set_default_dtype(ScalarType.BFloat16);


			Scalar stk = tokenclasses;
			//Scalar wordEmbeddingMomentum = 0.9;

			Tensor encoder = randn(tokenclasses, latentTokenSize, bfloat16, CUDA);
			using (Tensor tempmask = tensor(emasks, bfloat16, CUDA)) encoder.mul_(tempmask.unsqueeze_(1));

			encoder.requires_grad = true;
			Parameter decoder = nn.Parameter(randn(latentTokenSize, tokenclasses, bfloat16, CUDA).div_(Math.Sqrt(tokenclasses)));
			AdaBelief adaBelief = new AdaBelief(new Parameter[] { decoder }, 0.9, 0.999, 1e-9, 1e-15);


			Tensor encoderGrad = torch.zeros_like(encoder);
			encoder.set_grad(encoderGrad);

			Tensor decoderGrad = torch.zeros_like(decoder);
			decoder.set_grad(decoderGrad);

			Scalar slr = -Math.Sqrt(latentTokenSize);
			Scalar wd = 0.99;
			Scalar one = 1.0;



			ReadOnlyMemory<(ushort, double)>[] readOnlyMemories;
			using (BufferedStream dfs = new BufferedStream(new DeflateStream(new FileStream(datadir + "SimpleDecoder.model", FileMode.Open, FileAccess.Read, FileShare.Read, 16777216, FileOptions.SequentialScan), CompressionMode.Decompress), 16777216))
			{
				readOnlyMemories = Misc.LoadSimpleDecoderV2(tokenclasses, dfs);
			}








			Span<ushort> masked = stackalloc ushort[maxContextSize - 1];
			//Scalar costscale = costScalingTerm;



			Console.WriteLine("Waiting for dataset tokenization to complete...");
			foreach (Thread thr in thrlist)
			{
				thr.Join();
			}



			Console.WriteLine("Optimizing memory usage...");
			questionanswering = null;
			wikiarticles = null;
			dict1 = null;
			wqlen2 = wikisplit;
			ushort[][] tokenized = alldata.ToArray();



			int alldatasize = tokenized.Length;
			int wikidatasize = alldatasize - wqlen2;

			alldata = null;

			int maxgcgen = GC.MaxGeneration;
			GC.Collect(maxgcgen, GCCollectionMode.Aggressive, true, true);
			GC.WaitForPendingFinalizers();


			Console.WriteLine("Start training...");
			for (int z = 0; z < trainingBatches; ++z)
			{
				Console.WriteLine("Train batch #" + z);

				using IDisposable d2 = NewDisposeScope();
				int totalTokensGenerated = 0;
				int unlabeledTokensGenerated = 0;
				int labeledTokensGenerated = 0;
				double totalLosses = 0;
				int[] tokenCounts = new int[tokenclasses];

				while (true)
				{
					int mode;
					if (labeledTokensGenerated < targetLabeledTokensPerBatch)
					{
						mode = 0;
					}
					else if (unlabeledTokensGenerated < targetUnlabeledTokensPerBatch)
					{
						mode = 1;
					}
					else
					{
						break;
					}

					//FAST non-branching mode selector
					ushort[] example = tokenized[RandomNumberGenerator.GetInt32(wqlen2 * mode, wqlen2 + (wikidatasize * mode))];

					int lenm1 = example.Length;
					int split = example[0] + 1;

					int tokensGenerated = lenm1 - split;
					long[] cputarget2 = new long[tokensGenerated];
					for (int i = split; i < lenm1; ++i) cputarget2[i - split] = example[i];


					totalTokensGenerated += tokensGenerated;
					if (mode == 0)
					{
						labeledTokensGenerated += tokensGenerated;
					}
					else
					{
						unlabeledTokensGenerated += tokensGenerated;
					}
					split -= 2;
					ReadOnlySpan<ushort> spanny = example.AsSpan(1, lenm1 - 2);
					Dictionary<long, bool> mydict2 = new();
					for(int i = 0; ; ){
						ushort mv = spanny[i++];
						if (mv == 0) break;
						mydict2.TryAdd(mv, true);
					}

					using (NewDisposeScope())
					{

						long[] lar = mydict2.Keys.ToArray();
						for (int i = 0, stop = lar.Length; i < stop;) tokenCounts[lar[i++]] += tokensGenerated;
						
						Tensor x;
						using (Tensor y = tensor(lar, ScalarType.Int64, CUDA))
						{
							x = encoder[y];
						}
						using (Tensor y = x) x = y.sum(0,false);
						using (Tensor y = x) x = y.arcsinh();
						using (Tensor y = x) x = y.matmul(decoder);
						using (Tensor y = x) x = y.to(float32);

						float[,] floats = Misc.SimpleDecodePreLogV2Float(readOnlyMemories, spanny.Slice(split), true);

						using (Tensor y = x, z1 = tensor(floats, float32, CUDA))
						{
							z1.log_();
							x = y.add(z1);
						}


						using (Tensor target = tensor(cputarget2, ScalarType.Int64, CUDA), y = x)
						{
							x = Misc.FastCrossEntropyLoss(y, target, 0.0, false, null, 0.0, false, true);
						}
						using (x)
						{
							totalLosses += x.ToDouble();
							x.backward();
						}


					}
				}


				totalLosses /= totalTokensGenerated;



				Console.WriteLine("Average loss per token: " + totalLosses);

				Console.WriteLine("Scaling gradients...");
				decoderGrad.div_(totalTokensGenerated);


				Console.WriteLine("Applying regularization...");
				Misc.L2RegularizeIMPL(decoder, 1e-4);

				//notchatgpt.L1Regularize(0.1);
				Console.WriteLine("Optimizer step");
				
				adaBelief.Step(1e-4, false, false, 0.0);
				decoderGrad.zero_();
				using(no_grad()){
					
					using(Tensor zd = tensor(tokenCounts, bfloat16, CUDA)){
						zd.clamp_min_(one);
						zd.unsqueeze_(1);
						encoder.addcdiv_(encoderGrad,zd, slr);
					}
					encoder.mul_(wd);
				}
				encoderGrad.zero_();




				//if (z == 1024) return;




				GC.KeepAlive(d2);
			}
			Console.WriteLine("Saving policy...");
			new Core.SimpleAttention("",encoder, decoder).save(save);
		}
	}
}