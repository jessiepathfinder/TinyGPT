using Newtonsoft.Json;
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

namespace TinyGPT.DecoderV1.Trainer
{
	internal static class Program
	{


		//hyperparameters
		private const int latentTokenSize = 2048;
		private const int maxContextSize = 1025;
		private const int trainingBatches = 100000;
		private const int targetUnlabeledTokensPerBatch = 65536;
		private const int targetLabeledTokensPerBatch = 65536;
		private const int attentionHeads = 16;
		private const int firstTierAttentionDepth = 5;
		private const int magicTokenClasses = 4;
		private const int minimumInputTokens = 2;
		private const double regularizationTerm = 0.5;
		private const float firstOccouranceBoost = 1.5F;
		//private const int maxOutputBlockSize = 128;
		private const byte maskProbability = 32;
		private const byte randomRemoveProbability = 16;
		private const byte randomVerbCorruptionProbability = 32;
		private const int suboptimalSkipInitialTokens = 32;
		private const int regularizationLookback = 0;
		private const int startUnsupervisedTreshold = int.MaxValue;
		private const byte unsupervisedLearningRatio = 128;
		private const int wordEmbeddingUnlkTreshold = 256;
		private const int wordEmbeddingRelockTreshold = 512;
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
						if(ctd == 0){
							continue;
						}
						if (suboptimal) encsize2 += Math.Min(ctd - 1,suboptimalSkipInitialTokens);
						size1 += ctd;
						if (size1 < maxContextSize)
						{
							encbuffer2[size1++] = 1; //GPT-to-user context switch
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
							encbuffer[0] = (ushort)(size1);
							size1 += ctd;
							if (size1 < maxContextSize)
							{
								encbuffer2[size1++] = 1; //GPT-to-user context switch
							}

							


							alldata.Enqueue(encbuffer[..(size1 + 1)].ToArray());


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

			set_default_dtype(ScalarType.BFloat16);

			
			Scalar stk = tokenclasses;
			//Scalar wordEmbeddingMomentum = 0.9;

			double initstd = 1.0 / Math.Sqrt(latentTokenSize);

			GPTDecoderUnitV1_1 notchatgpt = new GPTDecoderUnitV1_1("TinyGPT", latentTokenSize, attentionHeads, firstTierAttentionDepth, initstd, 1e-7, 1024, initstd, 1.0, 1.0, 0.125, tokenclasses, 1.0, 128, 0.125, 1, 2048, 0.125,4);
			//Parameter hashedDecoderEngine = nn.Parameter(randn(latentTokenSize, latentTokenSize, ScalarType.BFloat16, CUDA).mul_(1.0 / Math.Sqrt(latentTokenSize)),true);

			//Dropout dropout = torch.nn.Dropout(0.25);

			notchatgpt.to(CUDA, ScalarType.BFloat16);

			//notchatgpt.supplementalWordEmbedding.mul_(0.02);



			IEnumerable <Parameter> parameters = notchatgpt.parameters();
			AdaBelief optimizer = new AdaBelief(parameters, 0.9, 0.999, 1e-9, 1e-15);
			//LRScheduler learningRateScheduler = ExponentialLR(adam, 0.9999, 0, true);



			notchatgpt.train(true);
			Span<ushort> masked = stackalloc ushort[maxContextSize - 1];
			//Scalar costscale = costScalingTerm;






			Queue<string> savequeue = new Queue<string>();
			long[] shape1 = new long[] { 1, -1 };


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

			//double alr = 1e-4;

			Console.WriteLine("Start training...");
			for (int z = 0; z < trainingBatches; ++z)
			{
				Console.WriteLine("Train batch #" + z);
				using IDisposable d2 = NewDisposeScope();
				int totalTokensGenerated = 0;
				int unlabeledTokensGenerated = 0;
				int labeledTokensGenerated = 0;
				double totalLosses = 0;
				/*
				if (z == wordEmbeddingUnlkTreshold)
				{
					Console.WriteLine("Unlocking word embeddings for training...");
					word2vec_weights.requires_grad_(true);
				}
				*/

				while (labeledTokensGenerated <= targetLabeledTokensPerBatch & unlabeledTokensGenerated <= targetUnlabeledTokensPerBatch)
				{
					int mode;
					if (labeledTokensGenerated > targetLabeledTokensPerBatch)
					{
						mode = 1;
					}
					else if (unlabeledTokensGenerated > targetUnlabeledTokensPerBatch)
					{
						mode = 0;
					}
					else
					{
						mode = RandomNumberGenerator.GetInt32(0, 2);
					}

					//FAST non-branching mode selector
					ushort[] example = tokenized[RandomNumberGenerator.GetInt32(wqlen2 * mode, wqlen2 + (wikidatasize * mode))];

					int lenm1 = example.Length;
					int split = example[0];
					
					Dictionary<ushort, bool> blacklist = new Dictionary<ushort, bool>();
					int newsplit = Transformer.MaskOrRamdomRemove(example.AsSpan(1, split - 1), masked, maskProbability, randomRemoveProbability, 3, blacklist);
					split += 1;

					int tokensGenerated = lenm1 - split;
					long[] cputarget2 = new long[tokensGenerated];
					for (int copy = split, c2 = 0; copy < lenm1; )
					{
						cputarget2[c2++] = example[copy++];
					}

					//bool allowUnsupervised = z > startUnsupervisedTreshold;







					Transformer.Mask(example.AsSpan(split - 1, tokensGenerated), masked.Slice(newsplit, tokensGenerated), maskProbability, 3, blacklist);
					split = newsplit;
					lenm1 = newsplit + tokensGenerated;

					if (z < 16) for (int i = 0, stop = tokensGenerated - 1; i < stop; ++i)
					{
						long data = masked[i + split + 1];
						if(data != 3 && data != cputarget2[i]){
							throw new Exception("Spans misaligned (should not reach here)");
						}


					}



					totalTokensGenerated += tokensGenerated;
					if (mode == 0)
					{
						labeledTokensGenerated += tokensGenerated;
					}
					else
					{
						unlabeledTokensGenerated += tokensGenerated;
					}

					using (NewDisposeScope())
					{

						Tensor x;
						using (Tensor y = notchatgpt.Forward(masked[..lenm1], split,0.0))
						{
							x = y.to(ScalarType.Float32);
						}


						using (Tensor y = x, target = tensor(cputarget2, ScalarType.Int64, CUDA))
						{
							x = Misc.FastCrossEntropyLoss(y, target, 0.025, false, null, 2, false);
						}
						
							
						using (x)
						{
							x.backward();
							totalLosses += x.ToDouble();
						}


					}
				}


				totalLosses /= totalTokensGenerated;

				Console.WriteLine("Average loss per token: " + totalLosses);




				Console.WriteLine("Scaling gradients...");
				Scalar costdiv = totalTokensGenerated;
				foreach (Parameter parameter in parameters)
				{
					(parameter.grad() ?? throw new Exception("Where is my grad? (should not reach here)")).div_(costdiv);
				}




				Console.WriteLine("Applying regularization...");
				//nn.utils.clip_grad_value_(parameters, 1);
				notchatgpt.L2Regularize(1e-4);
				notchatgpt.L1Regularize(1e-5);
				
				using (no_grad()){
					foreach (KeyValuePair<string, Tensor> kvp in notchatgpt.state_dict())
					{
						Tensor? t = kvp.Value.grad();
						if (t is null)
						{
							continue;
						}
						Tensor mean;
						Tensor std;
						Tensor min;
						Tensor max;
						using (Tensor t2 = t.to(ScalarType.Float32))
						{
							(std, mean) = t2.std_mean(false);
							min = t2.min();
							max = t2.max();
						}

						Console.WriteLine("{0}: std: {1} mean: {2} min: {3} max: {4}", kvp.Key.PadRight(25), std.ToSingle().ToString().PadRight(15), mean.ToSingle().ToString().PadRight(15), min.ToSingle().ToString().PadRight(15), max.ToSingle());
						std.Dispose();
						mean.Dispose();
						min.Dispose();
						max.Dispose();
					}
				}
				
				//notchatgpt.L1Regularize(0.1);
				Console.WriteLine("Optimizer step");
				optimizer.Step(1e-4, false, false, 0.0);
				optimizer.zero_grad();




				if (z % 128 == 0)
				{
					Console.WriteLine("Saving policy...");
					string savename = save + z;
					notchatgpt.save(savename);
					savequeue.Enqueue(savename);
					if (savequeue.Count > 5)
					{
						string name = savequeue.Dequeue();
						File.Delete(name);
					}

				}


				GC.KeepAlive(d2);
			}

		}
	}
}