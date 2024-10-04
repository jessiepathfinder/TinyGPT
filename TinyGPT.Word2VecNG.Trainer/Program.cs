using Newtonsoft.Json;
using static TorchSharp.torch;
using System.Collections.Concurrent;
using System.IO.Compression;
using System.Text;
using TinyGPT.Core;
using TorchSharp;
using System.Security.Cryptography;
using TorchSharp.Modules;
using Transformer = TinyGPT.Core.Transformer;

namespace TinyGPT.Word2VecNG.Trainer
{
	internal static class Program
	{


		//hyperparameters
		private const int latentTokenSize = 2048;
		private const int maxContextSize = 4096;
		private const int trainingBatches = 2048;
		private const int magicTokenClasses = 3;
		private const int minimumInputTokens = 2;
		private const int minContextSize = 16;

		private const int batchWidth = 2048;
		private const int batchHeight = 256;



		//private const double decoupledWeightDecay = 0.999;
		private const double lr1 = 224.1851913 / 2.0;
		private const double lr3 = 1e-4;
		private const double decoupledWeightDecay = 0.99;
		private const int batchWidthSkipGram = 512;
		private const int batchHeightSkipGram = 512;
		private const double skipGramLossAmplify = 10.0;
		private const double L2Regularization = 1e-4;
		//private const double decoupledWeightDecay = 0.99;


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
		private static void GlobalNorm(Tensor a, Tensor b, Tensor c)
		{
			Console.WriteLine("Global normalization...");
			Tensor y;
			using (Tensor x = a.transpose(0, 1)) y = x.matmul(b);
			using (Tensor x = y) y = x.transpose(0, 1);
			using (y) a.sub_(y);
			a.mul_(c);
			using (Tensor x = a.mul(a)) y = x.transpose(0, 1);
			using (Tensor x = y) y = x.matmul(b);
			y.sqrt_();
			using (Tensor x = y) y = x.transpose(0, 1);

			using (y) a.div_(y);

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

			ConcurrentBag<ushort[]>? alldata = new();
			//ConcurrentBag<int[]>? classcounters = new();
			int threads = Environment.ProcessorCount;
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
					Span<ushort> encbuffer2 = stackalloc ushort[maxContextSize];
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
						else if (size1 < minContextSize) continue;



						alldata.Add(encbuffer2[..(size1)].ToArray());


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
							size1 += ctd;
							if (size1 < maxContextSize)
							{
								encbuffer2[size1++] = 1; //GPT-to-user context switch
							}
							else if (size1 < minContextSize) continue;




							alldata.Add(encbuffer2[..(size1)].ToArray());


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



			Console.WriteLine("Initializing model...");
			InitializeDeviceType(DeviceType.CUDA);
			backends.cuda.matmul.allow_tf32 = true;
			backends.cuda.matmul.allow_fp16_reduced_precision_reduction = false;
			backends.cuda.enable_math_sdp(false);
			backends.cuda.enable_flash_sdp(true);
			backends.cudnn.allow_tf32 = true;
			Scalar one = 1.0;

			set_default_dtype(ScalarType.BFloat16);
			Scalar stc = Math.Sqrt(tokenclasses);

			Tensor model = randn(new long[] { tokenclasses, latentTokenSize }, ScalarType.BFloat16, CUDA);
			model.requires_grad = true;
			Parameter decoder = nn.Parameter(randn(new long[] { latentTokenSize, tokenclasses }, ScalarType.BFloat16, CUDA).div_(stc));

			Parameter skipGramWeight = nn.Parameter(randn(new long[] { latentTokenSize, tokenclasses }, ScalarType.BFloat16, CUDA).div_(stc));
			Parameter skipGramBias = nn.Parameter(zeros(tokenclasses, bfloat16, CUDA));/*.fill_(Math.Log((tokenclasses - 10) / 10.0)))*/
			//Parameter skipGramWeightFtr = nn.Parameter(randn(new long[] { latentTokenSize, tokenclasses }, ScalarType.BFloat16, CUDA).div_(stc));
			//Parameter skipGramBiasFtr = nn.Parameter(zeros(tokenclasses, bfloat16, CUDA));/*.fill_(Math.Log((tokenclasses - 10) / 10.0)))*/

			AdaBelief adabelief = new AdaBelief(new Parameter[] { skipGramWeight, decoder, skipGramBias }, 0.9, 0.999, 1e-9, 1e-15);



			Scalar wd = decoupledWeightDecay;
			long[] dimlist = new long[] { 0 };
			long[] labels = new long[batchWidth];
			//long[] skipGramInputs = new long[batchWidthSkipGram];
			long[,] inputs = new long[4, batchWidth];
			long[] skipGramInputs = new long[batchWidthSkipGram];
			Scalar epsilon = 1e-7;
			Scalar ld = batchWidth * batchHeight;
			Scalar slr1 = -lr1;
			Scalar lrs = L2Regularization;

			
			double tmd = 10.0 / (tokenclasses);
			double tmd2 = (1.0 / (1.0 - tmd));
			long tmd3 = (batchHeightSkipGram * ((long)batchWidthSkipGram) * (long)tokenclasses);
			Scalar ld2 = (tmd2 * skipGramLossAmplify) / tmd3;
			Scalar ld3 = (((1.0 / tmd) - tmd2) * skipGramLossAmplify) / (tmd3 * 2);
			

			//Scalar ld2 = -(batchWidthSkipGram * batchHeightSkipGram * 10);
			//Scalar swd = decoupledWeightDecay;

			Console.WriteLine("Waiting for dataset tokenization to complete...");
			foreach (Thread thr in thrlist)
			{
				thr.Join();
			}


			Console.WriteLine("Optimizing memory usage...");
			questionanswering = null;
			wikiarticles = null;
			dict1 = null;
			ushort[][] tokenized = alldata.ToArray();
			alldata = null;
			int alldatasize = tokenized.Length;
			int maxgcgen = GC.MaxGeneration;
			GC.Collect(maxgcgen, GCCollectionMode.Aggressive, true, true);
			GC.WaitForPendingFinalizers();

			Console.WriteLine("Constructing efficient sampling list...");
			(ushort[], int)[] tokenizedNG = new (ushort[], int)[alldatasize];
			for (int i = 0, acc = 0; i < alldatasize; ++i)
			{
				ushort[] us = tokenized[i];
				tokenizedNG[i] = (us, acc);
				acc += us.Length;
			}


			Console.WriteLine("Computing token probabilities...");
			int totalTokens = 0;
			int[] classcounts = new int[tokenclasses];
			//int[] classcountsd = new int[tokenclasses];
			for (int i = 0; i < alldatasize;)
			{
				ushort[] ushorts = tokenized[i++];
				int len = ushorts.Length;
				totalTokens += len;
				for (int y = 0; y < len;)
				{
					++classcounts[ushorts[y++]];
				}
				/*
				len -= 2;
				for (int y = 2; y < len;)
				{
					++classcountsd[ushorts[y++]];
				}
				*/
			}
			Tensor tClassCounts = tensor(classcounts, ScalarType.BFloat16, CUDA, false);
			//Tensor tClassCountsd = tensor(classcountsd, ScalarType.Float32, CUDA, false);
			//using (Tensor x = tClassCountsd) tClassCountsd = ((totalTokens - (alldatasize * 4)) / (double)(tokenclasses * ((long)batchWidth) * batchHeight)) / x;
			tClassCounts.unsqueeze_(1);
			Tensor normy;
			Scalar stt = totalTokens;
			Tensor normMask;
			using (no_grad())
			{
				normMask = tClassCounts.clamp_max(1);

				normy = tClassCounts.div(stt);

				GlobalNorm(model, normy, normMask);
			}
			tClassCounts.clamp_min_(one);
			tClassCounts.div_(stt);





			for (int z = 0; z < trainingBatches; ++z)
			{
				Console.WriteLine("Training batch #" + z);
				double totalLosses = 0.0;
				for (int y = 0; y < batchHeight; ++y)
				{
					


					for (int i = 0; i < batchWidth; ++i)
					{
						ReadOnlySpan<ushort> thething = GetSubSlice(tokenizedNG, totalTokens, 5);
						inputs[0, i] = thething[0];
						inputs[1, i] = thething[1];
						labels[i] = thething[2];
						inputs[2, i] = thething[3];
						inputs[3, i] = thething[4];
					}
					Tensor x;
					using (Tensor t = tensor(inputs, ScalarType.Int64, CUDA)) x = model[t];

					using (Tensor t = x) x = t.mean(dimlist, false);
					//using (Tensor t = x) x = t.arctan();
					//using (Tensor t = x) x = nn.functional.dropout(t, 0.25);
					using (Tensor t = x, t2 = randn_like(t)) x = t.add(t2);
					//using (Tensor t = x) x = t.to(float32);
					//using (Tensor t = x) x = CustomActivations.Norm(t, epsilon);
					//using (Tensor t = x) x = t.to(bfloat16);

					using (Tensor t = x) x = t.matmul(decoder);
					using (Tensor t = x) x = t.to(float32);
					using (Tensor t = x, t2 = tensor(labels, ScalarType.Int64, CUDA)) x = Misc.FastCrossEntropyLoss(t, t2, 0.0, false);
					using (Tensor t = x) x = t.div(ld);
					using (x)
					{
						x.backward();
						totalLosses += x.ToScalar().ToDouble();
					}
				}
				Console.WriteLine("Average CBOW training loss: " + totalLosses);



				totalLosses = 0.0;
				for (int y = 0; y < batchHeightSkipGram; ++y)
				{

					byte[,] mask = new byte[batchWidthSkipGram, tokenclasses];
					for (int i = 0; i < batchWidthSkipGram; ++i)
					{
						ReadOnlySpan<ushort> thething = GetSubSlice(tokenizedNG, totalTokens, 11);
						mask[i, thething[0]] = 2;
						mask[i, thething[1]] = 2;
						mask[i, thething[2]] = 2;
						mask[i, thething[3]] = 2;
						mask[i, thething[4]] = 2;



						skipGramInputs[i] = thething[5];
						mask[i, thething[6]] = 2;
						mask[i, thething[7]] = 2;
						mask[i, thething[8]] = 2;
						mask[i, thething[9]] = 2;
						mask[i, thething[10]] = 2;
					}

					Tensor x;
					using (Tensor t = tensor(skipGramInputs, ScalarType.Int64, CUDA)) x = model[t];
					using (Tensor t = x, t2 = randn_like(t)) x = t.add(t2);
					//
					//using (Tensor t = x) x = nn.functional.dropout(x, 0.5);
					//using (Tensor t = x) x = t.to(float32);
					//using (Tensor t = x) x = CustomActivations.Norm(t, epsilon);
					//using (Tensor t = x) x = CustomActivations.HalfNorm(t);
					//using (Tensor t = x) x = t.to(bfloat16);
					using (Tensor t = x) x = t.matmul(skipGramWeight);
					using (Tensor t = x) x = t.add(skipGramBias);
					Tensor t3;
					using (Tensor t2 = tensor(mask, ScalarType.BFloat16, CUDA))
					{
						t3 = empty_like(t2).fill_(ld2);
						t3.add_(t2, ld3);
						using Tensor t = x;
						x = t.mul(t2.sub_(one));
					}

					using (Tensor t = x) x = t.softplus();
					using (t3)
					{
						using Tensor t = x;
						x = t.mul(t3);
					}
					using (Tensor t = x) x = t.sum(ScalarType.Float64);
					using (x)
					{
						x.backward();
						totalLosses += x.ToScalar().ToDouble();
					}
				}

				Console.WriteLine("Average skip-gram training loss: " + totalLosses);



				Console.WriteLine("L2 regularization...");
				using (no_grad())
				{
					Misc.L2RegularizeIMPL(decoder, lrs);
					Misc.L2RegularizeIMPL(skipGramWeight, lrs);
				}

				Console.WriteLine("Optimizer step (Decoder)...");
				using (no_grad())
				{
					adabelief.Step(lr3, false, false, 0.0);
					adabelief.zero_grad();
					Console.WriteLine("Optimizer step (Encoder)...");
					Tensor mgrad = model.grad() ?? throw new Exception("Model does not have grad (should not reach here)");
					model.addcdiv_(mgrad, tClassCounts, slr1);
					model.mul_(wd);
					mgrad.zero_();
				}

			}
			using (no_grad()) GlobalNorm(model, normy, normMask);
			Console.WriteLine("Saving policy...");
			model.save(save);
		}
		private static ReadOnlySpan<ushort> GetSubSlice((ushort[] arr, int i)[] ushorts,int total, int size){
			int i = 0;
#pragma warning disable CS8619 // Nullability of reference types in value doesn't match target type.
			(ushort[],int) xr = (null,RandomNumberGenerator.GetInt32(0, total));
#pragma warning restore CS8619 // Nullability of reference types in value doesn't match target type.

			int ind = Array.BinarySearch(ushorts, xr, TheComparer.instance);

			if (ind < 0) ind = Math.Max((~ind) - 1, 0);

			(ushort[] data,_) = ushorts[ind];


			return data.AsSpan(RandomNumberGenerator.GetInt32(0, data.Length - size), size);
		}
		private sealed class TheComparer : IComparer<(ushort[] arr, int i)>
		{
			private TheComparer() { }
			public static readonly TheComparer instance = new TheComparer();
			public int Compare((ushort[] arr, int i) x, (ushort[] arr, int i) y)
			{
				return x.i.CompareTo(y.i);
			}
		}
	}
}