using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System.Collections.Concurrent;
using System.ComponentModel;
using System.ComponentModel.Design;
using System.IO.Compression;
using System.Linq.Expressions;
using System.Runtime.CompilerServices;
using System.Runtime.ExceptionServices;
using System.Security.AccessControl;
using System.Security.Cryptography;
using System.Text;
using System.Xml.Schema;
using TinyGPT.Core;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.distributions.constraints;
using static TorchSharp.torch.optim;
using static TorchSharp.torch.optim.lr_scheduler;
using Transformer = TinyGPT.Core.Transformer;

namespace TinyGPT.DecoderV1.Trainer
{
	internal static class Program
	{


		//hyperparameters
		private const int latentTokenSize = 512;
		private const int maxContextSize = 512;
		private const int trainingBatches = 2000000;
		private const int targetUnlabeledTokensPerBatch = 1024;
		private const int targetLabeledTokensPerBatch = 1024;
		private const int attentionHeads = 8;
		private const int firstTierAttentionDepth = 1;
		private const int magicTokenClasses = 5;
		private const int minimumInputTokens = 5;
		private const double regularizationTerm = 0.5;
		private const double firstOccouranceBoost = 1.5;
		//private const int maxOutputBlockSize = 128;
		private const byte maskProbability = 24;
		private const byte randomRemoveProbability = 12;
		private const byte randomVerbCorruptionProbability = 32;
		private const int suboptimalSkipInitialTokens = 80;
		private const int regularizationLookback = 70;
		private const byte unsupervisedLearningRatio = 128;
		private const int startUnsupervisedTreshold = 512;
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
			//[START_GPT], [END_GPT], [WIKI_SEPERATOR], [MASK], [INVALID_COPY]
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
			string progresstail = new StringBuilder("/").Append(wqlen2).Append(" question-answer pairs").ToString();
			string wikiprogresstail = new StringBuilder("/").Append(wikilen2).Append(" simple english wikipedia articles").ToString();

			
			for (int z = 0; z < threads; ++z)
			{
				int az = z;
				Thread thread = new Thread(() =>
				{
					int za = az;
					StringBuilder sb = new StringBuilder("Tokenized ");
					Span<ushort> encbuffer = stackalloc ushort[maxContextSize + 2];
					Span<ushort> encbuffer2 = encbuffer.Slice(2, maxContextSize);
					int mywqlen = wqlength;
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
						ushort truesize = (ushort)(size1 + 2);
						int encsize2 = size1 + (suboptimal ? suboptimalSkipInitialTokens : 0);


						encbuffer[size1++] = 0; //user-to-GPT context switch
						if (size1 == maxContextSize)
						{
							continue;
						}
						//int encsize2 = size1;
						int ctd = Transformer.Tokenize(dict1, encbuffer2[size1..], pair[1], maxlen, magicTokenClasses);
						if (ctd < minimumInputTokens | (suboptimal & ctd <= suboptimalSkipInitialTokens))
						{
							continue;
						}
						size1 += ctd;
						if (size1 < maxContextSize)
						{
							encbuffer2[size1++] = 1; //GPT-to-user context switch
						}

						encbuffer[0] = (ushort)(encsize2 + 2);
						encbuffer[1] = truesize;


						alldata.Enqueue(encbuffer[..(size1 + 2)].ToArray());


						if ((a & 4095) == 4095)
						{
							Console.WriteLine(sb.Append(a).Append(progresstail).ToString());
							sb.Remove(10, sb.Length - 10);
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
							int encsize2 = size1;
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

							ushort ssize = (ushort)(encsize2 + 2);
							encbuffer[0] = ssize;
							encbuffer[1] = ssize;


							alldata.Enqueue(encbuffer[..(size1 + 2)].ToArray());


						}
						if ((a & 4095) == 4095)
						{
							Console.WriteLine(sb.Append(a).Append(wikiprogresstail).ToString());
							sb.Remove(10, sb.Length - 10);
						}
					}
				});
				thread.Name = "Dataset tokenizer thread #" + z;
				thread.IsBackground = true;
				thrlist[z] = thread;
				thread.Start();
			}
			string[] tokens = new string[tokenclasses];
			foreach (KeyValuePair<string, ushort> kvp in dict)
			{
				tokens[kvp.Value + magicTokenClasses] = kvp.Key;
			}



			Console.WriteLine("Initializing model...");
			InitializeDeviceType(DeviceType.CUDA);
			backends.cuda.matmul.allow_tf32 = true;
			backends.cuda.matmul.allow_fp16_reduced_precision_reduction = false;
			backends.cuda.enable_math_sdp(false);
			backends.cuda.enable_flash_sdp(true);
			backends.cudnn.allow_tf32 = true;
			GPTDecoderUnitV1 notchatgpt = new GPTDecoderUnitV1("TinyGPT", latentTokenSize, attentionHeads, tokenclasses, firstTierAttentionDepth, 0.5, latentTokenSize, attentionHeads, 1e-7, 4, 1, 2048);

			//Dropout dropout = torch.nn.Dropout(0.25);

			notchatgpt.to(CUDA, ScalarType.BFloat16);
			Linear? currentTokenEngine = Misc.CreateKaimingInitializedLinear(latentTokenSize * attentionHeads, latentTokenSize, false, nn.init.FanInOut.FanIn);
			currentTokenEngine.to(CUDA, ScalarType.BFloat16);
			IEnumerable<Parameter> parameters = notchatgpt.parameters();
			IEnumerable<Parameter> effectiveParams = Misc.JoinEnumerators(parameters, currentTokenEngine.weight ?? throw new Exception("Current token engine does not have weights (should not reach here)"));
			AdaBelief adabelief = new AdaBelief(effectiveParams, 0.9, 0.999, 1e-6);
			List<ushort> options = new List<ushort>();
			//LRScheduler learningRateScheduler = ExponentialLR(adam, 0.9999, 0, true);
			Span<byte> rsb = stackalloc byte[2048];
			int randctr = 0;


			notchatgpt.train(true);
			Span<ushort> masked = stackalloc ushort[maxContextSize];
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


			Console.WriteLine("Start training...");
			double adaptiveLearningRate = 1e-4;
			for (int z = 0; z < trainingBatches; ++z)
			{
				Console.WriteLine("Train batch #" + z);
				bool allowUnsupervised = z > startUnsupervisedTreshold & currentTokenEngine is null;
				using IDisposable d2 = NewDisposeScope();
				int totalTokensGenerated = 0;
				int unlabeledTokensGenerated = 0;
				int labeledTokensGenerated = 0;

				double totalLosses = 0;


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
						mode = RandomNumberGenerator.GetInt32(0, 1);
					}

					//FAST non-branching mode selector
					ushort[] example = tokenized[RandomNumberGenerator.GetInt32(wqlen2 * mode, wqlen2 + (wikidatasize * mode))];

					int lenm1 = example.Length;
					int split = example[0];
					Dictionary<ushort, bool> boostdict = new Dictionary<ushort, bool>();
					int truestart = example[1];
					while(truestart < split){
						boostdict.TryAdd(example[truestart++], false);
					}

					long[] cputarget2 = new long[lenm1 - split];
					double[] boostvector = new double[lenm1 - split];


					if(allowUnsupervised){
						for (int copy = split, c2 = 0; copy < lenm1; ++c2)
						{
							ushort data = example[copy++];
							long data1 = data;
							double efb;
							if (boostdict.TryAdd(data, false))
							{
								efb = firstOccouranceBoost;
								data1 += 1;
							}
							else
							{
								efb = 1.0;
								if (randctr == 2048)
								{
									RandomNumberGenerator.Fill(rsb);
									randctr = 0;
								}
								if (rsb[randctr++] > unsupervisedLearningRatio)
								{
									data1 += 1;
								} else{
									data1 = 0;
								}
							}

							cputarget2[c2] = data1;
							boostvector[c2] = efb;
						}
					} else{
						for (int copy = split, c2 = 0; copy < lenm1; ++c2)
						{
							ushort data = example[copy++];
							cputarget2[c2] = data;
							boostvector[c2] = (boostdict.TryAdd(data, false) ? firstOccouranceBoost : 1.0);
						}
					}
					split -= 3;
					lenm1 -= 3;
					Dictionary<ushort, bool> blacklist = new Dictionary<ushort, bool>();
					int newsplit = Transformer.MaskOrRamdomRemove(example.AsSpan(2, split), masked, maskProbability, randomRemoveProbability, 3, blacklist);
					Transformer.Mask(example.AsSpan(2 + split, lenm1 - split), masked[newsplit..], maskProbability, 3, blacklist);
					lenm1 -= split - newsplit;
					split = newsplit;
					
					for (int copy = 0; copy < lenm1; ++copy)
					{
						if(randctr == 2048){
							RandomNumberGenerator.Fill(rsb);
							randctr = 0;
						}
						if (rsb[randctr++] > randomVerbCorruptionProbability)
						{
							continue;
						}

						ref ushort token = ref masked[copy];

						string? value = tokens[token];

						if (value is null)
						{
							continue;
						}
						if (value.EndsWith("ing"))
						{
							value = value.Substring(0, value.Length - 3);
							if (dict.TryGetValue(value, out ushort tkn))
							{
								options.Add(tkn);
							}
							if (dict.TryGetValue(value + "ed", out tkn))
							{
								options.Add(tkn);
							}
							if (dict.TryGetValue(value + "s", out tkn))
							{
								options.Add(tkn);
							}
							if (dict.TryGetValue(value + "d", out tkn))
							{
								options.Add(tkn);
							}
							if (dict.TryGetValue(value + "en", out tkn))
							{
								options.Add(tkn);
							}
						}
						else if (value.EndsWith("ed"))
						{
							value = value.Substring(0, value.Length - 2);
							if (dict.TryGetValue(value, out ushort tkn))
							{
								options.Add(tkn);
							}
							if (dict.TryGetValue(value + "ing", out tkn))
							{
								options.Add(tkn);
							}
							if (dict.TryGetValue(value + "s", out tkn))
							{
								options.Add(tkn);
							}
							if (dict.TryGetValue(value + "d", out tkn))
							{
								options.Add(tkn);
							}
							if (dict.TryGetValue(value + "en", out tkn))
							{
								options.Add(tkn);
							}
						}
						else if (value.EndsWith("en"))
						{
							value = value.Substring(0, value.Length - 2);
							if (dict.TryGetValue(value, out ushort tkn))
							{
								options.Add(tkn);
							}
							if (dict.TryGetValue(value + "ing", out tkn))
							{
								options.Add(tkn);
							}
							if (dict.TryGetValue(value + "s", out tkn))
							{
								options.Add(tkn);
							}
							if (dict.TryGetValue(value + "d", out tkn))
							{
								options.Add(tkn);
							}
							if (dict.TryGetValue(value + "ed", out tkn))
							{
								options.Add(tkn);
							}
						}
						else if (value.EndsWith('s'))
						{
							value = value.Substring(0, value.Length - 1);
							if (dict.TryGetValue(value, out ushort tkn))
							{
								options.Add(tkn);
							}
							if (dict.TryGetValue(value + "ing", out tkn))
							{
								options.Add(tkn);
							}
							if (dict.TryGetValue(value + "ed", out tkn))
							{
								options.Add(tkn);
							}
							if (dict.TryGetValue(value + "d", out tkn))
							{
								options.Add(tkn);
							}
							if (dict.TryGetValue(value + "en", out tkn))
							{
								options.Add(tkn);
							}
						}
						else if (value.EndsWith('d'))
						{
							value = value.Substring(0, value.Length - 1);
							if (dict.TryGetValue(value, out ushort tkn))
							{
								options.Add(tkn);
							}
							if (dict.TryGetValue(value + "ing", out tkn))
							{
								options.Add(tkn);
							}
							if (dict.TryGetValue(value + "ed", out tkn))
							{
								options.Add(tkn);
							}
							if (dict.TryGetValue(value + "s", out tkn))
							{
								options.Add(tkn);
							}
							if (dict.TryGetValue(value + "en", out tkn))
							{
								options.Add(tkn);
							}
						}
						else
						{
							if (dict.TryGetValue(value + "en", out ushort tkn))
							{
								options.Add(tkn);
							}
							if (dict.TryGetValue(value + "ing", out tkn))
							{
								options.Add(tkn);
							}
							if (dict.TryGetValue(value + "ed", out tkn))
							{
								options.Add(tkn);
							}
							if (dict.TryGetValue(value + "d", out tkn))
							{
								options.Add(tkn);
							}
							if (dict.TryGetValue(value + "en", out tkn))
							{
								options.Add(tkn);
							}
						}
						string val2 = value.ToLower();

						if (val2 == value)
						{
							val2 = (value.Length > 1) ? (char.ToUpper(value[0]) + value.Substring(1)) : value.ToUpper();
						}
						if (val2 != value)
						{
							if (dict.TryGetValue(val2, out ushort tkn))
							{
								options.Add(tkn);
							}
							if (dict.TryGetValue(val2 + "en", out tkn))
							{
								options.Add(tkn);
							}
							if (dict.TryGetValue(val2 + "ing", out tkn))
							{
								options.Add(tkn);
							}
							if (dict.TryGetValue(val2 + "ed", out tkn))
							{
								options.Add(tkn);
							}
							if (dict.TryGetValue(value + "d", out tkn))
							{
								options.Add(tkn);
							}
							if (dict.TryGetValue(val2 + "en", out tkn))
							{
								options.Add(tkn);
							}
						}
						int oc = options.Count;
						if (oc == 0)
						{
							continue;
						}
						token = (ushort)(options[oc == 1 ? 0 : RandomNumberGenerator.GetInt32(0, oc)] + magicTokenClasses);
						options.Clear();
					}
					int tokensGenerated = lenm1 - split;
					
					
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

						Tensor encoded = notchatgpt.Encode(masked[..lenm1], split, 0.0);

						Tensor x;
						using (Tensor y = notchatgpt.DefaultDecode(encoded))
						{
							x = y.to(ScalarType.Float32);
						}



						Tensor loss;
						using (Tensor y = x, target = tensor(cputarget2, ScalarType.Int64, CUDA), boost = tensor(boostvector, ScalarType.Float32, CUDA))
						{
							loss = Misc.FastCrossEntropyLoss(y, target, 0.025, false, boost, 2, allowUnsupervised);
						}




						if (currentTokenEngine is null)
						{
							encoded.Dispose();
						}
						else
						{
							long[] cputarget = new long[lenm1 - split];
							for (int copy = split; copy < lenm1; ++copy)
							{
								cputarget[copy - split] = masked[copy];
							}
							using (Tensor y = encoded)
							{
								x = currentTokenEngine.forward(y);
							}

							using (Tensor y = x)
							{
								x = notchatgpt.Decode(y);
							}
							using (Tensor y = x)
							{
								x = y.to(ScalarType.Float64);
							}



							using (Tensor y = x, logits = tensor(cputarget, ScalarType.Int64, CUDA))
							{
								x = Misc.FastCrossEntropyLoss(y, logits, 0.025, false, null, 2, false);
							}

							using (Tensor y = loss)
							{
								using (x)
								{
									loss = y.add(x, regularizationTerm);
								}
							}
						}
						/*
						using(Tensor y = loss){
							loss = y.div(costscale);
						}
						*/
						using (loss)
						{
							loss.backward();
							totalLosses += loss.ToDouble();
						}

					}
				}


				totalLosses /= totalTokensGenerated;

				Console.WriteLine("Average loss per token: " + totalLosses);

				double alr2 = (adaptiveLearningRate * 0.999) + (totalLosses * 4e-9);
				if (alr2 < adaptiveLearningRate)
				{
					Console.WriteLine("Setting adaptive learning rate to " + alr2);
					adaptiveLearningRate = alr2;
				}

				if (currentTokenEngine is { } && totalLosses < 16.5)
				{
					Console.WriteLine("Activating terminal mode...");
					currentTokenEngine.Dispose();
					currentTokenEngine = null;
					effectiveParams = parameters;
					adabelief.EraseInvalids();
				}
				Console.WriteLine("Scaling gradients...");
				Scalar costdiv = totalTokensGenerated;
				foreach (Parameter parameter in effectiveParams)
				{
					(parameter.grad() ?? throw new Exception("Where is my grad? (should not reach here)")).div_(costdiv);
				}

				Console.WriteLine("Applying regularization...");
				//nn.utils.clip_grad_value_(parameters, 1);
				notchatgpt.L2Regularize(0.01);
				//notchatgpt.L1Regularize(0.1);




				Console.WriteLine("Optimizer step");
				adabelief.Step(adaptiveLearningRate);
				adabelief.zero_grad();

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