using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System.Collections.Concurrent;
using System.ComponentModel;
using System.ComponentModel.Design;
using System.Net.Http.Headers;
using System.Runtime.ExceptionServices;
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
		private const int latentTokenSize = 1024;
		private const int maxContextSize = 1024;
		private const int trainingBatches = 100000;
		private const int targetTokensPerHugeBatch = 65536;
		private const int targetTokensPerSmallBatch = 4096;
		private const int attentionHeads = 12;
		private const int firstTierAttentionDepth = 8;
		private const int magicTokenClasses = 4;
		private const int hugeBatchTransition = 5000;

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

			//2 magic token types
			//[START_GPT], [END_GPT]
			tokenclasses += magicTokenClasses + 1;
			Console.WriteLine("Loading ELI5 + WikiQA question answering dataset...");
			string[][]? questionanswering = JsonConvert.DeserializeObject<string[][]>(File.ReadAllText(datadir + "QuestionAnswering.json"));
			if (questionanswering is null)
			{
				Console.WriteLine("Null question answering dataset");
				return;
			}
			int wqlen2 = questionanswering.Length;

			Console.WriteLine("Loading simple english wikipedia dataset...");
			string[]? wikiarticles = File.ReadAllLines(datadir + "simplewiki-latest.jsonl");


			int wikilen2 = wikiarticles.Length;

			Console.WriteLine("Starting dataset tokenizers...");
			int wqlength = wqlen2;
			int wikilen = wikilen2;

			ConcurrentQueue<(Tensor, ushort[])>? alldata = new();
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
					Span<ushort> encbuffer = stackalloc ushort[maxContextSize + 1];
					Span<ushort> encbuffer2 = encbuffer.Slice(1, maxContextSize);
					int mywqlen = wqlength;
					int mywikilen = wikilen;
					int[][]? sizeclasses = new int[maxContextSize][];
					for (int i = 1; i < maxContextSize; ++i)
					{
						sizeclasses[i] = new int[i];
					}
					while (true)
					{
						int a = Interlocked.Increment(ref loadprogress);
						if (a > mywqlen)
						{
							break;
						}
						a -= 1;
						string[] pair = questionanswering[a];


						int size1 = Transformer.Tokenize(dict, encbuffer2, pair[0], maxlen, magicTokenClasses);
						if (size1 == maxContextSize)
						{
							continue;
						}
						ushort encsize = (ushort)size1;
						

						encbuffer[size1++] = 0; //user-to-GPT context switch
						if (size1 == maxContextSize)
						{
							continue;
						}
						int encsize2 = size1;
						int ctd = Transformer.Tokenize(dict, encbuffer2[size1..], pair[1], maxlen, magicTokenClasses);
						if (ctd == 0)
						{
							continue;
						}
						size1 += ctd;
						if (size1 < maxContextSize)
						{
							encbuffer2[size1++] = 1; //GPT-to-user context switch

						}
						encbuffer[0] = encsize;
						int[] convert = sizeclasses[size1 - encsize2];
						for (int i = encsize2; i < size1; ++i)
						{
							convert[i - encsize2] = encbuffer2[i];
						}

						
						alldata.Enqueue((tensor(convert, ScalarType.Int64, CUDA), encbuffer[..(size1 + 1)].ToArray()));
						

						if ((a & 4095) == 4095)
						{
							Console.WriteLine(sb.Append(a).Append(progresstail).ToString());
							sb.Remove(10, sb.Length - 10);
						}

					}

					barrier3.SignalAndWait();
					if(za == 0){
						wikisplit = alldata.Count;
					}
					barrier3.SignalAndWait();
					if(za == 0){
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
						int size2 = Transformer.Tokenize(dict, encbuffer2, wikititle, maxlen, magicTokenClasses);
						if (size2 == maxContextSize)
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
							int size1 = Transformer.Tokenize(dict, encbuffer2[size2..], title, maxlen, magicTokenClasses) + size2;
							if (size1 == maxContextSize)
							{
								continue;
							}


							ushort encsize = (ushort)size1;
							encbuffer2[size1++] = 0; //[START_GPT]
							if (size1 == maxContextSize)
							{
								continue;
							}
							int encsize2 = size1;
							int ctd = Transformer.Tokenize(dict, encbuffer2[size1..], text.Replace("'''", null).Replace("''", null), maxlen, magicTokenClasses);
							if (ctd == 0)
							{
								continue;
							}
							size1 += ctd;
							if (size1 < maxContextSize)
							{
								encbuffer2[size1++] = 1; //GPT-to-user context switch

							}
							encbuffer[0] = encsize;
							int[] convert = sizeclasses[size1 - encsize2];
							for (int i = encsize2; i < size1; ++i)
							{
								convert[i - encsize2] = encbuffer2[i];
							}
							

							alldata.Enqueue((tensor(convert, ScalarType.Int64, CUDA), encbuffer[..(size1 + 1)].ToArray()));

							
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




			Console.WriteLine("Initializing model...");
			InitializeDeviceType(DeviceType.CUDA);
			GPTDecoderUnitV1 notchatgpt = new GPTDecoderUnitV1("TinyGPT", latentTokenSize, attentionHeads, tokenclasses, firstTierAttentionDepth, 0.25, 512, 512, 2048, 1e-7);
			notchatgpt.to(CUDA, ScalarType.BFloat16);
			Adam adam = new Adam(notchatgpt.parameters(), lr: 1e-4, eps: 1e-7);
			//LRScheduler learningRateScheduler = ExponentialLR(adam, 0.9999, 0, true);
			NLLLoss crossEntropyLoss = new NLLLoss(reduction: nn.Reduction.Sum);

			adam.to(CUDA);

			notchatgpt.train(true);







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
			wqlen2 = wikisplit;
			(Tensor, ushort[])[] tokenized = alldata.ToArray();

			int alldatasize = tokenized.Length;
			int wikidatasize = alldatasize - wqlen2;

			alldata = null;

			GC.Collect(GC.MaxGeneration, GCCollectionMode.Aggressive, true, true);
			GC.WaitForPendingFinalizers();

			Span<ushort> masked = stackalloc ushort[maxContextSize];
			Console.WriteLine("Start multithreaded forward workers...");





			Console.WriteLine("Start training...");
			double adaptiveLearningRate = 1e-5;

			for (int z = 0; z < trainingBatches; ++z)
			{
				Console.WriteLine("Forward pass batch #" + z);
				adam.zero_grad();
				using var d2 = NewDisposeScope();
				int totalTokensGenerated = 0;
				double totalLosses = 0;
				bool small = z < hugeBatchTransition;
				int currentTarget = small ? targetTokensPerSmallBatch : targetTokensPerHugeBatch;
				while (totalTokensGenerated < currentTarget)
				{
					//FAST non-branching mode selector
					int mode = small ? 1 : RandomNumberGenerator.GetInt32(0, 2);
					(Tensor target, ushort[] example) = tokenized[RandomNumberGenerator.GetInt32(wqlen2 * mode, wqlen2 + (wikidatasize * mode))];

					int lenm1 = example.Length - 2;

					Transformer.Mask(example.AsSpan(1, lenm1), masked, 8, 3, new Dictionary<ushort, bool>());
					int split = example[0];
					totalTokensGenerated += lenm1 - split;


					using (NewDisposeScope())
					{


						
						
						Tensor x;
						using (Tensor onehot1 = notchatgpt.Forward(masked[..lenm1], split))
						{
							x = onehot1.to(ScalarType.Float64);
						}
						using (Tensor y = x)
						{
							x = y.log_softmax(1);
						}
						using (Tensor y = x)
						{
							x = crossEntropyLoss.forward(y, target);
						}
						
						Tensor cpuloss;

						using (x)
						{
							x.backward();
							cpuloss = x.cpu();
						}
						using (cpuloss)
						{
							totalLosses += cpuloss.ToDouble();
						}


					}
				}



				totalLosses /= totalTokensGenerated;

				Console.WriteLine("Average loss per token: " + totalLosses);



				adaptiveLearningRate = (adaptiveLearningRate * 0.999) + (Math.Min(totalLosses, 10) * 1e-8);

				Console.WriteLine("scaling gradients...");
				foreach (Tensor tensor in notchatgpt.parameters())
				{
					(tensor.grad() ?? throw new Exception("Unexpected null gradients (should not reach here)")).div_(totalTokensGenerated);
				}

				Console.WriteLine("Applying regularization...");
				notchatgpt.L1Regularize(0.001);

				Console.WriteLine("setting adaptive learning rate to " + adaptiveLearningRate);
				foreach (ILearningRateController learningRateController in adam.ParamGroups)
				{
					learningRateController.LearningRate = adaptiveLearningRate;
				}



				Console.WriteLine("Optimizer step");

				//learningRateScheduler.step();
				adam.step();
				if ((!small) & (z % 32 == 0))
				{
					Console.WriteLine("Saving policy...");
					string savename = save + z;
					notchatgpt.save(savename);
					savequeue.Enqueue(savename);
					if (savequeue.Count > 5)
					{
						File.Delete(savequeue.Dequeue());
					}
				}
			}

		}
	}
}