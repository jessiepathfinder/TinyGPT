using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System.Collections.Concurrent;
using System.ComponentModel;
using System.ComponentModel.Design;
using System.Drawing;
using System.Net.Http.Headers;
using System.Runtime.ExceptionServices;
using System.Runtime.Intrinsics.X86;
using System.Security.Cryptography;
using System.Text;
using System.Xml.Schema;
using TinyGPT.Core;
using TorchSharp;
using TorchSharp.Modules;
using static System.Net.Mime.MediaTypeNames;
using static TorchSharp.torch;
using static TorchSharp.torch.distributions.constraints;
using static TorchSharp.torch.optim;
using static TorchSharp.torch.optim.lr_scheduler;
using Transformer = TinyGPT.Core.Transformer;

namespace TinyGPT.DecoderV1.Trainer
{
	internal static class Program
	{
		private sealed class DecoderBlocksLinkedList
		{
			public readonly DecoderBlocksLinkedList? next;
			public readonly ushort[] data;

			public DecoderBlocksLinkedList(DecoderBlocksLinkedList? next, ushort[] data)
			{
				this.next = next;
				this.data = data;
			}
		}

		//hyperparameters
		private const int latentTokenSize = 512;
		private const int maxContextSize = 512;
		private const int trainingBatches = 500000;
		private const int trainingMicroBatchSize = 16;
		private const int attentionHeads = 8;
		private const int firstTierAttentionDepth = 16;
		private const int totalMagicTokens = 4;
		private const int optimizerUpdateInterval = 4;
		private const int targetTokensPerBatch = 2048;

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

			//[START_GPT], [END_GPT]
			tokenclasses += totalMagicTokens + 1;
			Console.WriteLine("Loading ELI5 + WikiQA question answering dataset...");
			string[][]? questionanswering = JsonConvert.DeserializeObject<string[][]>(File.ReadAllText(datadir + "QuestionAnswering.json"));
			if (questionanswering is null)
			{
				Console.WriteLine("Null question answering dataset");
				return;
			}

			Console.WriteLine("Loading simple english wikipedia dataset...");

			ConcurrentQueue<(int start, DecoderBlocksLinkedList list)>? alldata = new();
			string[]? wikiarticles = File.ReadAllLines(datadir + "simplewiki-latest.jsonl");
			int wikilen = wikiarticles.Length;
			int wqlength = questionanswering.Length;


			Console.WriteLine("Starting dataset tokenizers...");

			int threads = Environment.ProcessorCount;
			int loadprogress = 0;
			int wikiloadprogress = 0;
			Thread[] thrlist = new Thread[threads];
			int wqlen2 = wqlength;
			string progresstail = new StringBuilder("/").Append(wqlen2).Append(" question-answer pairs").ToString();
			string wikiprogresstail = new StringBuilder("/").Append(wikilen).Append(" simple english wikipedia articles").ToString();
			Barrier? barrier3 = new Barrier(threads);
			int seperatorLen = 0;
			for (int z = 0; z < threads; ++z)
			{
				int az = z;
				Thread thread = new Thread(() =>
				{
					StringBuilder sb = new StringBuilder("Tokenized ");
					Span<ushort> encbuffer2 = stackalloc ushort[maxContextSize];
					int stopsize = wqlength;
					//int stopsize = 256;
					Stack<ushort[]> stackbuilder = new Stack<ushort[]>();
					while (true)
					{
						int a = Interlocked.Increment(ref loadprogress);
						if (a > stopsize)
						{
							break;
						}
						a -= 1;
						string[] pair = questionanswering[a];


						int size1 = Transformer.Tokenize(dict, encbuffer2, pair[0], maxlen, totalMagicTokens, out _);
						if (size1 == maxContextSize)
						{
							continue;
						}
						int encsize = size1;

						encbuffer2[size1++] = 0; //user-to-GPT context switch
						if (size1 == maxContextSize)
						{
							continue;
						}
						string ostr = pair[1];

						int size4 = size1;
						bool contribute = false;
						int slicestr = 0;
						ushort[]? prevblk = null;
					encloop:
						int newtokens = Transformer.Tokenize(dict, encbuffer2[size4..], ostr.AsSpan(slicestr), maxlen, totalMagicTokens, out int slicestr1);
						slicestr += slicestr1;
						size4 += newtokens;

						int modsize = size4 % maxContextSize;
						if (modsize == 0)
						{
							if (newtokens > 0)
							{
								prevblk = encbuffer2.ToArray();
								stackbuilder.Push(prevblk);
								contribute = true;
								size4 = 1;
								encbuffer2[0] = encbuffer2[maxContextSize - 1];
								goto encloop;
							}
						}
						else
						{
							encbuffer2[modsize++] = 1; //GPT-to-user context switch

							if (newtokens > 0)
							{
								contribute = true;
								stackbuilder.Push(encbuffer2[..(modsize)].ToArray());
							}
							else if (prevblk is { })
							{
								prevblk[maxContextSize - 1] = 1; //[END_GPT]
							}
						}


						DecoderBlocksLinkedList? decoderBlocks = null;
#pragma warning disable CS8600 // Converting null literal or possible null value to non-nullable type.
						while (stackbuilder.TryPop(out ushort[] array))
						{
							if (array.Length == 1)
							{
								Console.WriteLine("WARNING: array of length 1 found");
								stackbuilder.Clear();
								break;
							}
							decoderBlocks = new DecoderBlocksLinkedList(decoderBlocks, array);
						}
#pragma warning restore CS8600 // Converting null literal or possible null value to non-nullable type.
						if (decoderBlocks is { } && contribute)
						{
							alldata.Enqueue((encsize, decoderBlocks));
						}
						else
						{
							Console.WriteLine("WARNING: empty output found");
						}


						if ((a & 4095) == 4095)
						{
							Console.WriteLine(sb.Append(a).Append(progresstail).ToString());
							sb.Remove(10, sb.Length - 10);
						}

					}
					barrier3.SignalAndWait();
					if (az == 0)
					{
						seperatorLen = alldata.Count;
					}
					barrier3.SignalAndWait();
					if (az == 0)
					{
						barrier3.Dispose();
						barrier3 = null;
					}
					stopsize = wikilen;
					while (true)
					{
						int a = Interlocked.Increment(ref wikiloadprogress);
						if (a > stopsize)
						{
							return;
						}
						a -= 1;
						WikipediaArticle? wikipediaArticle = JsonConvert.DeserializeObject<WikipediaArticle>(wikiarticles[a]);
						if (wikipediaArticle is null)
						{
							goto continue2;
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
						int size2 = Transformer.Tokenize(dict, encbuffer2, wikititle, maxlen, totalMagicTokens, out _);
						if (size2 == maxContextSize)
						{
							goto continue2;
						}

						encbuffer2[size2++] = 2; //wikipedia article retrieval task

						if (size2 == maxContextSize)
						{
							goto continue2;
						}
						Span<ushort> encbuffer3 = encbuffer2[size2..];
						int stop = maxContextSize - size2;

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
							int size1 = Transformer.Tokenize(dict, encbuffer3, title, maxlen, totalMagicTokens, out _);
							if (size1 == stop)
							{
								continue;
							}

							encbuffer3[size1++] = 0; //[START_GPT]
							if (size1 == stop)
							{
								continue;
							}
							size1 += size2;
							int size4 = size1;
							int encsize = size1;
							bool contribute = false;
							int slicestr = 0;
							ushort[]? prevblk = null;
						encloop:
							int newtokens = Transformer.Tokenize(dict, encbuffer2[size4..], text.AsSpan(slicestr), maxlen, totalMagicTokens, out int slicestr1);
							slicestr += slicestr1;
							size4 += newtokens;

							int modsize = size4 % maxContextSize;
							if (modsize == 0)
							{
								if (newtokens > 0)
								{
									prevblk = encbuffer2.ToArray();
									stackbuilder.Push(prevblk);
									contribute = true;
									size4 = 1;
									encbuffer2[0] = encbuffer2[maxContextSize - 1];
									goto encloop;
								}
							}
							else
							{
								encbuffer2[modsize++] = 1; //GPT-to-user context switch

								if (newtokens > 0)
								{
									contribute = true;
									stackbuilder.Push(encbuffer2[..(modsize)].ToArray());
								}
								else if (prevblk is { })
								{
									prevblk[maxContextSize - 1] = 1; //[END_GPT]
								}
							}
							DecoderBlocksLinkedList? decoderBlocks = null;
#pragma warning disable CS8600 // Converting null literal or possible null value to non-nullable type.
							while (stackbuilder.TryPop(out ushort[] array))
							{
								if (array.Length == 1)
								{
									Console.WriteLine("WARNING: array of length 1 found");
									stackbuilder.Clear();
									break;
								}
								decoderBlocks = new DecoderBlocksLinkedList(decoderBlocks, array);
							}
#pragma warning restore CS8600 // Converting null literal or possible null value to non-nullable type.
							if (decoderBlocks is { } && contribute)
							{
								alldata.Enqueue((encsize, decoderBlocks));
							}
							else
							{
								Console.WriteLine("WARNING: empty output found");
							}

						}


					continue2:
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
			int[][] sizeclasses = new int[maxContextSize][];
			for (int i = 1; i < maxContextSize; ++i)
			{
				sizeclasses[i - 1] = new int[i];
			}


			Console.WriteLine("Initializing model...");
			InitializeDeviceType(DeviceType.CUDA);
			GPTDecoderUnitV1 notchatgpt = new GPTDecoderUnitV1("TinyGPT", latentTokenSize, attentionHeads, tokenclasses, firstTierAttentionDepth, 0.25, 1e-8);
			notchatgpt.to(CUDA, ScalarType.BFloat16);
			Adam adam = new Adam(notchatgpt.parameters(), lr: 1e-5, eps: 1e-8);
			//LRScheduler learningRateScheduler = ExponentialLR(adam, 0.9999, 0, true);

			NLLLoss nlloss = new NLLLoss(reduction: nn.Reduction.Sum);

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
			wqlen2 = seperatorLen;
			(int size, DecoderBlocksLinkedList decoderBlocksLinkedList)[] tokenized = alldata.ToArray();

			int alldatasize = tokenized.Length;
			int wikidatasize = alldatasize - wqlen2;

			alldata = null;

			GC.Collect(GC.MaxGeneration, GCCollectionMode.Aggressive, true, true);
			GC.WaitForPendingFinalizers();

			Span<ushort> masked = stackalloc ushort[maxContextSize];


			Console.WriteLine("Start training...");
			double adaptiveLearningRate = 1e-5;
			for (int z = 0, savecooldown = 0; z < trainingBatches; ++z)
			{
				Console.WriteLine("Forward pass batch #" + z);
				using var d2 = NewDisposeScope();
				int totalTokensGenerated = 0;
				double totalLosses = 0;
				while (true)
				{


					DecoderBlocksLinkedList? decoderBlocksLinkedList = null;
					Tensor? memory = null;
					Dictionary<ushort, bool> state = new Dictionary<ushort, bool>();
					int slice = 0;


					if (decoderBlocksLinkedList is null)
					{
						if (totalTokensGenerated > targetTokensPerBatch)
						{
							break;
						}
						//FAST non-branching mode selector
						int mode = RandomNumberGenerator.GetInt32(0, 2);
						(slice, decoderBlocksLinkedList) = tokenized[RandomNumberGenerator.GetInt32(wqlen2 * mode, wqlen2 + (wikidatasize * mode))];
						if (decoderBlocksLinkedList is null)
						{
							throw new Exception("Unexpected null linked list root (should not reach here)");
						}
						state.Clear();
					}
					ushort[] example = decoderBlocksLinkedList.data;
					int lenm1 = example.Length - 1;

					Transformer.Mask(example.AsSpan(0, lenm1), masked, 8, 3, state);


					using (NewDisposeScope())
					{

						using Tensor? memory2 = memory;
						using Tensor memory1 = notchatgpt.Encode(masked[..(lenm1)], memory2);
						lenm1 -= slice;
						int[] convert = sizeclasses[lenm1 - 1];
						for (int ci = 0; ci < lenm1;)
						{
							ref int destination = ref convert[ci];
							destination = example[++ci + slice];
						}
						totalTokensGenerated += lenm1;
						Tensor loss;
						Tensor onehot1;
						using (Tensor onehot = notchatgpt.Decode(memory1, slice, memory2))
						{
							onehot1 = onehot.to(ScalarType.Float64);
						}
						using (Tensor onehot = onehot1)
						{
							onehot1 = onehot.log_softmax(1);
						}
						using (Tensor logits = tensor(convert, ScalarType.Int64, CUDA))
						{
							using (onehot1)
							{
								loss = nlloss.forward(onehot1, logits);
							}
						}
						Tensor cpuloss;
						using (loss)
						{
							loss.backward();
							cpuloss = loss.cpu();
						}
						using (cpuloss)
						{
							totalLosses += cpuloss.ToDouble();
						}



						decoderBlocksLinkedList = decoderBlocksLinkedList.next;
						if (decoderBlocksLinkedList is null)
						{
							memory = null;
						}
						else
						{
							memory = memory1.detach().MoveToOuterDisposeScope();
						}


					}

					slice = 0;
				}



				totalLosses /= totalTokensGenerated;

				Console.WriteLine("Average loss per token: " + totalLosses);



				adaptiveLearningRate = (adaptiveLearningRate * 0.99) + (Math.Min(totalLosses, 10) * 1e-7);

				Console.WriteLine("scaling gradients...");
				foreach (Tensor tensor in notchatgpt.parameters())
				{
					(tensor.grad() ?? throw new Exception("Unexpected null gradients (should not reach here)")).div_(totalTokensGenerated);
				}
				Console.WriteLine("setting adaptive learning rate to " + adaptiveLearningRate);
				foreach (ILearningRateController learningRateController in adam.ParamGroups)
				{
					learningRateController.LearningRate = adaptiveLearningRate;
				}


				if (++savecooldown == 256)
				{
					Console.WriteLine("Saving policy...");
					string savename = save + z;
					notchatgpt.save(savename);
					savequeue.Enqueue(savename);
					if (savequeue.Count > 5)
					{
						File.Delete(savequeue.Dequeue());
					}
					savecooldown = 0;
				}
				Console.WriteLine("Optimizer step");

				//learningRateScheduler.step();
				adam.step();
			}

		}
	}
}