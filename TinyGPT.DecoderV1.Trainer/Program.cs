using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System.Collections.Concurrent;
using System.ComponentModel;
using System.ComponentModel.Design;
using System.Drawing;
using System.Net.Http.Headers;
using System.Runtime.ExceptionServices;
using System.Security.Cryptography;
using System.Text;
using System.Xml.Schema;
using TinyGPT.Core;
using TorchSharp;
using TorchSharp.Modules;
using static System.Net.Mime.MediaTypeNames;
using static TorchSharp.torch;
using static TorchSharp.torch.optim;
using static TorchSharp.torch.optim.lr_scheduler;
using Transformer = TinyGPT.Core.Transformer;

namespace TinyGPT.DecoderV1.Trainer
{
	internal static class Program
	{
		private sealed class DecoderBlocksLinkedList{
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
		private const int firstTierAttentionDepth = 5;
		private const int totalMagicTokens = 4;
		private const int optimizerUpdateInterval = 3;

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

			ConcurrentBag<(int start, DecoderBlocksLinkedList list)>? alldata = new();
			ConcurrentBag<(int start, DecoderBlocksLinkedList list)>? safedata = new();
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
			int safeSeperatorLen = 0;
			for (int z = 0; z < threads; ++z)
			{
				int az = z;
				Thread thread = new Thread(() =>
				{
					StringBuilder sb = new StringBuilder("Tokenized ");
					Span<ushort> encbuffer2 = stackalloc ushort[maxContextSize];
					int stopsize = wqlength;
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

						bool noninitial = false;
						bool contribute = false;
						int slicestr = 0;
					encloop:
						int newtokens = Transformer.Tokenize(dict, noninitial ? encbuffer2 : encbuffer2[size1..], ostr.AsSpan(slicestr), maxlen, totalMagicTokens, out int slicestr1);
						slicestr += slicestr1;
						size1 += newtokens;

						int modsize = size1 % maxContextSize;
						if (modsize == 0 & newtokens > 0)
						{
							stackbuilder.Push(encbuffer2.ToArray());
							contribute = true;
							noninitial = true;
							goto encloop;
						}


						if (modsize > 0)
						{
							encbuffer2[modsize++] = 1; //GPT-to-user context switch
						}
						else
						{
							encbuffer2[maxContextSize - 1] = 1; //GPT-to-user context switch
						}
						if (newtokens > 0)
						{
							contribute = true;
							stackbuilder.Push(encbuffer2[..(modsize)].ToArray());
						}


						DecoderBlocksLinkedList? decoderBlocks = null;
#pragma warning disable CS8600 // Converting null literal or possible null value to non-nullable type.
						while (stackbuilder.TryPop(out ushort[] array)){
							if(array.Length == 1){
								Console.WriteLine("WARNING: array of length 1 found");
								break;
							}
							decoderBlocks = new DecoderBlocksLinkedList(decoderBlocks, array);
						}
#pragma warning restore CS8600 // Converting null literal or possible null value to non-nullable type.
						if(decoderBlocks is { } && contribute)
						{
							alldata.Add((encsize, decoderBlocks));
							if(noninitial){
								safedata.Add((encsize, decoderBlocks));
							}
						} else{
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
						safeSeperatorLen = safedata.Count;
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
							int encsize = size1;
							bool noninitial = false;
							bool contribute = false;
							int slicestr = 0;
						encloop:
							int newtokens = Transformer.Tokenize(dict, noninitial ? encbuffer2 : encbuffer2[size1..], text.AsSpan(slicestr), maxlen, totalMagicTokens, out int slicestr1);
							slicestr += slicestr1;
							size1 += newtokens;

							int modsize = size1 % maxContextSize;
							if (modsize == 0 & newtokens > 0)
							{
								stackbuilder.Push(encbuffer2.ToArray());
								noninitial = true;
								contribute = true;
								goto encloop;
							}


							if (modsize > 0)
							{
								encbuffer2[modsize++] = 1; //GPT-to-user context switch
							}
							else
							{
								encbuffer2[maxContextSize - 1] = 1; //GPT-to-user context switch
							}
							if (newtokens > 0)
							{
								contribute = true;
								stackbuilder.Push(encbuffer2[..(modsize)].ToArray());
							}
							DecoderBlocksLinkedList? decoderBlocks = null;
#pragma warning disable CS8600 // Converting null literal or possible null value to non-nullable type.
							while (stackbuilder.TryPop(out ushort[] array))
							{
								if (array.Length == 1)
								{
									Console.WriteLine("WARNING: array of length 1 found");
									break;
								}
								decoderBlocks = new DecoderBlocksLinkedList(decoderBlocks, array);
							}
#pragma warning restore CS8600 // Converting null literal or possible null value to non-nullable type.
							if (decoderBlocks is { } && contribute)
							{
								alldata.Add((encsize, decoderBlocks));
								if (noninitial)
								{
									safedata.Add((encsize, decoderBlocks));
								}
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




			Console.WriteLine("Initializing model...");
			InitializeDeviceType(DeviceType.CUDA);
			GPTDecoderUnitV1 notchatgpt = new GPTDecoderUnitV1("TinyGPT", latentTokenSize, attentionHeads, tokenclasses, firstTierAttentionDepth, 0.25, 1e-8);
			notchatgpt.to(CUDA, ScalarType.BFloat16);
			Adam adam = new Adam(notchatgpt.parameters(), lr: 1e-5, eps: 1e-8);
			//LRScheduler learningRateScheduler = ExponentialLR(adam, 0.9999, 0, true);

			NLLLoss crossEntropyLoss = new NLLLoss(reduction: nn.Reduction.Mean);

			adam.to(CUDA);

			notchatgpt.train(true);





			Barrier barrier1 = new Barrier(trainingMicroBatchSize + 1);
			Barrier barrier2 = new Barrier(trainingMicroBatchSize + 1);
			Tensor[] actualTensors = new Tensor[trainingMicroBatchSize];
			int[] expslices = new int[trainingMicroBatchSize];
			ushort[][] expectedValues = new ushort[trainingMicroBatchSize][];

			double bestloss = double.PositiveInfinity;
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
			int wqlen3 = safeSeperatorLen;
			(int size, DecoderBlocksLinkedList decoderBlocksLinkedList)[] safetokenized = safedata.ToArray();

			int alldatasize = tokenized.Length;
			int safedatasize = safetokenized.Length;
			alldata = null;
			safedata = null;

			GC.Collect(GC.MaxGeneration, GCCollectionMode.Aggressive, true, true);
			GC.WaitForPendingFinalizers();
			Console.WriteLine("Start multithreaded forward workers...");
			bool safepoint = false; //Safepoint tells batches to stop at a short data 
			int safevotes = 0;

			for (int i = 0; i < trainingMicroBatchSize; ++i)
			{
				int z = i;

				Thread thread = new Thread(() => {
					Span<ushort> masked = stackalloc ushort[maxContextSize];
					int z2 = z;
					bool mode = (z2 & 1) == 1;
					DecoderBlocksLinkedList? decoderBlocksLinkedList = null;
					Tensor? memory = null;
					Dictionary<ushort, bool> state = new Dictionary<ushort, bool>();
					int slice = 0;
					while (true)
					{
						barrier1.SignalAndWait();
						bool safepoint1 = safepoint;
						if (decoderBlocksLinkedList is null){
							(slice, decoderBlocksLinkedList) = safepoint1 ? safetokenized[mode ? RandomNumberGenerator.GetInt32(wqlen3, safedatasize) : RandomNumberGenerator.GetInt32(0, wqlen3)] : tokenized[mode ? RandomNumberGenerator.GetInt32(wqlen2, alldatasize) : RandomNumberGenerator.GetInt32(0, wqlen2)];
							if(decoderBlocksLinkedList is null){
								throw new Exception("Unexpected null linked list root (should not reach here)");
							}
							state.Clear();
						}
						ushort[] example = decoderBlocksLinkedList.data;
						Transformer.Mask(example, masked, 8, 3, state);

						expectedValues[z2] = example;


						using (NewDisposeScope())
						{
							Tensor memory1 = notchatgpt.Encode(masked[..(example.Length - 1)], memory);

							using(memory){
								actualTensors[z2] = notchatgpt.Decode(memory1, slice, memory).MoveToOuterDisposeScope();
							}
							decoderBlocksLinkedList = decoderBlocksLinkedList.next;
							if(decoderBlocksLinkedList is null){
								memory = null;

								//Cast vote saying that the optimizer can be safely ran
								if(safepoint1){
									Interlocked.Increment(ref safevotes);
								}
							} else{
								memory1.detach();
								memory = memory1.MoveToOuterDisposeScope();
							}
							
						}
						expslices[z2] = slice;
						slice = 0;
						barrier2.SignalAndWait();
					}
				});
				thread.IsBackground = true;
				thread.Name = "Forward thread #" + i;
				thread.Start();
			}




			Console.WriteLine("Start training...");
			double adaptiveLearningRate = 1e-4;
			for (int z = 0, savecooldown = 0, safecounter = 0; z < trainingBatches; ++z)
			{
				Console.WriteLine("Forward pass batch #" + z);
				using var d2 = NewDisposeScope();
				bool safepoint1 = safecounter > optimizerUpdateInterval;
				safepoint = safepoint1;
				if(++safecounter == 1){
					adam.zero_grad();
				}

				Tensor loss;
				
				barrier1.SignalAndWait();
				barrier2.SignalAndWait();

				int totallen = 0;
				for (int p = 0; p < trainingMicroBatchSize; ++p)
				{
					ushort[] arr = expectedValues[p];
					totallen += arr.Length - (expslices[p] + 1);
				}
				int[] ec2 = new int[totallen];
				for (int p = 0, pos = 0; p < trainingMicroBatchSize; ++p)
				{
					ushort[] arr = expectedValues[p];
					int arrlen = arr.Length;

					for (int pos2 = expslices[p] + 1; pos2 < arrlen;)
					{
						ec2[pos++] = arr[pos2++];
					}

				}
				Console.WriteLine("Compute loss batch #" + z);
				using (NewDisposeScope())
				{
					Tensor lsp;
					using (Tensor cat2 = cat(actualTensors, 0))
					{
						using Tensor catfloat = cat2.to(ScalarType.Float64);

						lsp = catfloat.log_softmax(1);

					}
					Tensor logits = tensor(ec2, ScalarType.Int64, CUDA);

					loss = crossEntropyLoss.forward(lsp, logits);

					Console.WriteLine("Backpropagate batch #" + z);
					using (NewDisposeScope())
					{
						loss.backward();
					}

					for (int p = 0; p < trainingMicroBatchSize; ++p)
					{
						actualTensors[p].Dispose();
					}
					loss = loss.cpu().MoveToOuterDisposeScope();
				}





				double totalloss2 = loss.ToDouble();
				Console.WriteLine("Batch loss: " + totalloss2);
				adaptiveLearningRate = (adaptiveLearningRate * 0.999) + (Math.Min(totalloss2, 10) * 1e-8);
				
				if(safepoint1)
				{
					int safevotes1 = safevotes;
					Console.WriteLine("Safepoint votes: " + safevotes1);

					//policy can only be updated at safepoints!!!
					//(because of the TinyGPT-XL recurrent transformer layers)
					if (safevotes1 == trainingMicroBatchSize){
						
						Console.WriteLine("scaling gradients...");
						foreach (Tensor tensor in notchatgpt.parameters())
						{
							(tensor.grad() ?? throw new Exception("Unexpected null gradients (should not reach here)")).div_(safecounter);
						}
						safecounter = 0;
						Console.WriteLine("setting adaptive learning rate to " + adaptiveLearningRate);
						foreach (ILearningRateController learningRateController in adam.ParamGroups)
						{
							learningRateController.LearningRate = adaptiveLearningRate;
						}


						bool defeat = totalloss2 < bestloss;
						if (++savecooldown == 64)
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
					safevotes = 0;
				}
			}

		}
	}
}