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
using static TorchSharp.torch.optim;
using static TorchSharp.torch.optim.lr_scheduler;
using Transformer = TinyGPT.Core.Transformer;

namespace TinyGPT.DecoderV1.Trainer
{
	internal static class Program
	{
#pragma warning disable CS8618 // Non-nullable field must contain a non-null value when exiting constructor. Consider declaring as nullable.
#pragma warning disable CS0649
		[JsonObject(MemberSerialization.Fields, MissingMemberHandling = MissingMemberHandling.Ignore)]
		private sealed class SquadDataset
		{

			public SquadTopic[] data;

		}
		[JsonObject(MemberSerialization.Fields, MissingMemberHandling = MissingMemberHandling.Ignore)]
		private sealed class SquadTopic
		{
			public SquadParagraph[] paragraphs;
		}
		[JsonObject(MemberSerialization.Fields, MissingMemberHandling = MissingMemberHandling.Ignore)]
		private sealed class SquadParagraph
		{
			public string context;
			public SquadQuestion[] qas;
		}
		[JsonObject(MemberSerialization.Fields, MissingMemberHandling = MissingMemberHandling.Ignore)]
		private sealed class SquadQuestion
		{
			public string question;
			public bool is_impossible;
			public SquadAnswer[] plausible_answers;
			public SquadAnswer[] answers;
		}
		[JsonObject(MemberSerialization.Fields, MissingMemberHandling = MissingMemberHandling.Ignore)]
		private sealed class SquadAnswer
		{
			public string text;

		}
#pragma warning disable CS0649
#pragma warning restore CS8618 // Non-nullable field must contain a non-null value when exiting constructor. Consider declaring as nullable.

		//hyperparameters
		private const int latentTokenSize = 512;
		private const int maxContextSize = 2048;
		private const int trainingBatches = 300000;
		private const int trainingMicroBatchSize = 16;
		private const int attentionHeads = 8;
		const int secondTierAttentionDepth = 5;
		private const int compressedViewSize = 1024;
		const int firstTierAttentionDepth = 3;




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

			//4 magic token types
			//[START_GPT], [END_GPT], [START_SQUAD_QUESTION], [SQUAD_IMPOSSIBLE]
			tokenclasses += 5;
			Console.WriteLine("Loading ELI5 + WikiQA question answering dataset...");
			string[][]? questionanswering = JsonConvert.DeserializeObject<string[][]>(File.ReadAllText(datadir + "QuestionAnswering.json"));
			Console.WriteLine("Loading SQuAD v2.0 dataset...");
			if (questionanswering is null)
			{
				Console.WriteLine("Null question answering dataset");
				return;
			}
			int wqlength = questionanswering.Length;

			SquadTopic[] squadTopics;
			{
				SquadDataset? squadroot = JsonConvert.DeserializeObject<SquadDataset>(File.ReadAllText(datadir + "squad-v2.0.json"));


				if (squadroot is null)
				{
					Console.WriteLine("Null SQuAD v2.0 dataset");
					return;
				}
				squadTopics = squadroot.data;
			}
			Console.WriteLine("Starting dataset tokenizers...");

			ConcurrentBag<(ReadOnlyMemory<ushort> context, ReadOnlyMemory<ushort> question_answer, ushort split)> squadQAItems = new ConcurrentBag<(ReadOnlyMemory<ushort> question_answer, ReadOnlyMemory<ushort> question, ushort split)>();
			ushort[][] tokenized = new ushort[wqlength][];
			int threads = Environment.ProcessorCount;
			int loadprogress = 0;
			int squadloadprogress = 0;
			Thread[] thrlist = new Thread[threads];
			int wqlen2 = wqlength;
			StringBuilder tempbuilder = new StringBuilder("/");
			string progresstail = tempbuilder.Append(wqlen2).Append(" question-answer pairs").ToString();
			int sqlen = squadTopics.Length;
			string progresstail2 = tempbuilder.Remove(1, progresstail.Length - 1).Append(sqlen).Append(" SQuAD v2.0 topics").ToString();
			int[] classcounter = new int[tokenclasses];
			int[] squadclasscounter = new int[tokenclasses];

			for (int z = 0; z < threads; ++z)
			{
				int az = z;
				Thread thread = new Thread(() =>
				{
					StringBuilder sb = new StringBuilder("Tokenized ");
					Span<ushort> encbuffer = stackalloc ushort[maxContextSize + 1];
					Span<ushort> encbuffer2 = encbuffer.Slice(1, maxContextSize);
					while (true)
					{
						int a = Interlocked.Increment(ref loadprogress);
						if (a > wqlength)
						{
							break;
						}
						a -= 1;
						string[] pair = questionanswering[a];


						int size1 = Transformer.Tokenize(dict, encbuffer2, pair[0], maxlen, 4);
						ushort encsize = (ushort)Math.Min(size1 + 1, maxContextSize - 1);
						int encsize2 = size1;
						if (size1 == maxContextSize)
						{
							goto flush;
						}
						encbuffer[size1++] = 0; //user-to-GPT context switch
						if (size1 == maxContextSize)
						{
							goto flush;
						}
						size1 += Transformer.Tokenize(dict, encbuffer2[size1..], pair[1], maxlen, 4);
						if (size1 == maxContextSize)
						{
							goto flush;
						}
						encbuffer[size1++] = 1; //GPT-to-user context switch

						//Memory regularization
						int copysize = Math.Min(maxContextSize - size1, encsize2);
						if (copysize > 0)
						{
							encbuffer[..copysize].CopyTo(encbuffer[size1..]);
							size1 += copysize;

						}
					flush:
						++size1;
						for(int rdx = encsize2 + 2; rdx < size1; ++rdx){
							Interlocked.Increment(ref classcounter[encbuffer[rdx]]);
						}
						encbuffer[0] = encsize;
						tokenized[a] = encbuffer[..(size1)].ToArray();

						if ((a & 4095) == 4095)
						{
							Console.WriteLine(sb.Append(a).Append(progresstail).ToString());
							sb.Remove(10, sb.Length - 10);
						}

					}

					encbuffer2 = encbuffer.Slice(0, maxContextSize - 1);
					while (true)
					{
						int a = Interlocked.Increment(ref squadloadprogress);
						if (a > sqlen)
						{
							break;
						}
						a -= 1;
						SquadParagraph[] topic = squadTopics[a].paragraphs;
						int topicount = topic.Length;
						for (int tpi = 0; tpi < topicount; ++tpi)
						{
							SquadParagraph squadParagraph = topic[tpi];
							int size1a = Transformer.Tokenize(dict, encbuffer2, squadParagraph.context, maxlen, 4);
							if (size1a == maxContextSize)
							{
								continue;
							}
							encbuffer[size1a++] = 2; //[START_SQUAD_QUESTION]
							if (size1a == maxContextSize)
							{
								continue;
							}

							ushort[] context = encbuffer[..size1a].ToArray();
							int maxsize1 = maxContextSize - size1a;
							encbuffer2 = encbuffer[..maxsize1];
							SquadQuestion[] squadQuestions = squadParagraph.qas;
							int sqlen3 = squadQuestions.Length;
							for (int sqi = 0; sqi < sqlen3; ++sqi)
							{
								SquadQuestion squadQuestion = squadQuestions[sqi];
								bool impossible = squadQuestion.is_impossible;
								SquadAnswer[] ansarr = impossible ? squadQuestion.plausible_answers : squadQuestion.answers;
								if (ansarr.Length == 0)
								{
									continue;
								}
								int size2 = Transformer.Tokenize(dict, encbuffer2, squadQuestion.question, maxlen, 4);
								if (size2 == maxsize1)
								{
									continue;
								}
								encbuffer[size2++] = 0; //[START_GPT]
								
								if (size2 == maxsize1)
								{
									continue;
								}
								int split = size2;

								if (impossible)
								{
									encbuffer[size2++] = 3; //[SQUAD_IMPOSSIBLE]
									if (size2 == maxsize1)
									{
										continue;
									}
								}
								size2 += Transformer.Tokenize(dict, encbuffer2[size2..], ansarr[0].text, maxlen, 4);
								if (size2 == maxsize1)
								{
									continue;
								}
								encbuffer[size2++] = 1; //[END_GPT]
								for(int rbx = split; rbx < size2; ++rbx) {
									Interlocked.Increment(ref squadclasscounter[encbuffer[rbx]]);
								}
								squadQAItems.Add((context, encbuffer[..size2].ToArray(), (ushort)(split + size1a)));
							}
						}

						if ((a & 15) == 15)
						{
							Console.WriteLine(sb.Append(a).Append(progresstail2).ToString());
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
			ModuleList<BERTDictionaryItem> dictionaryItems = new ModuleList<BERTDictionaryItem>();
			for (int i = 0; i < tokenclasses; ++i)
			{
				dictionaryItems.Add(new BERTDictionaryItem("", latentTokenSize));
			}
			GPTDecoderUnitV1 notchatgpt = new GPTDecoderUnitV1("TinyGPT", latentTokenSize, attentionHeads, tokenclasses, compressedViewSize, firstTierAttentionDepth, secondTierAttentionDepth, 1e-9);
			SimpleFullGPTDecoderUnit simpleFullGPTDecoderUnit = new SimpleFullGPTDecoderUnit(dictionaryItems, notchatgpt, "");

			simpleFullGPTDecoderUnit.to(CUDA, ScalarType.BFloat16);
			Adam adam2 = new Adam(dictionaryItems.parameters(), lr: 1e-5, 0.95, amsgrad: true, eps: 1e-9);
			Adam adam = new Adam(notchatgpt.parameters(), lr: 1e-5, 0.95, weight_decay: 1e-5, eps: 1e-9);
			LRScheduler learningRateScheduler = ExponentialLR(adam, 0.9998, 0, true);
			LRScheduler learningRateScheduler2 = ExponentialLR(adam2, 0.9998, 0, false);

			adam.to(CUDA);
			adam2.to(CUDA);
			
			notchatgpt.train(true);

			
			
			

			
			Barrier barrier1 = new Barrier(threads + 1);
			Barrier barrier2 = new Barrier(threads + 1);
			Tensor[] actualTensors = new Tensor[trainingMicroBatchSize];
			ushort[][] expectedClasses = new ushort[trainingMicroBatchSize][];
			
			double bestloss = double.PositiveInfinity;
			Queue<string> savequeue = new Queue<string>();
			long[] shape1 = new long[] { 1, -1 };

			Console.WriteLine("Waiting for dataset tokenization to complete...");
			foreach (Thread thr in thrlist)
			{
				thr.Join();
			}
			Console.WriteLine("Computing class weights...");
			Tensor classweights;
			using (NewDisposeScope())
			{
				Tensor cw1 = tensor(classcounter, ScalarType.Int32, CUDA).minimum(1);
				cw1 = cw1.sum().to(ScalarType.Float64).div(cw1.to(ScalarType.Float64));
				Tensor cw2 = tensor(squadclasscounter, ScalarType.Int32, CUDA).minimum(1);
				cw2 = cw2.sum().to(ScalarType.Float64).div(cw2.to(ScalarType.Float64));
				classweights = cw1.add_(cw2).div(2).MoveToOuterDisposeScope();
			}
			Console.WriteLine("Optimizing memory usage...");
			(ReadOnlyMemory<ushort> context, ReadOnlyMemory<ushort> question, ushort suplit)[] squadQAItemsArray = squadQAItems.ToArray();
			int sqalen2 = squadQAItemsArray.Length;

#pragma warning disable CS8600 // Converting null literal or possible null value to non-nullable type.
			squadQAItems = null;
			squadTopics = null;
			classcounter = null;
#pragma warning restore CS8600 // Converting null literal or possible null value to non-nullable type.
			questionanswering = null;

			GC.Collect(GC.MaxGeneration, GCCollectionMode.Aggressive, true, true);
			GC.WaitForPendingFinalizers();
			Console.WriteLine("Start multithreaded forward workers...");

			for (int i = 0; i < threads; ++i)
			{
				int z = i;
				Thread thread = new Thread(() => {
					while (true)
					{
						barrier1.SignalAndWait();
						for (int k = z; k < trainingMicroBatchSize; k += threads)
						{
							ushort[] example;
							if (RandomNumberGenerator.GetInt32(0, 2) == 0)
							{
								example = tokenized[RandomNumberGenerator.GetInt32(0, wqlen2)];
							}
							else
							{
								(ReadOnlyMemory<ushort> context, ReadOnlyMemory<ushort> question_answer, ushort split2) = squadQAItemsArray[RandomNumberGenerator.GetInt32(0, sqalen2)];
								int len1 = context.Length;
								int len2 = question_answer.Length;
								example = new ushort[len1 + len2 + 1];
								context.CopyTo(example.AsMemory(1));
								question_answer.CopyTo(example.AsMemory(1 + len1));
								example[0] = split2;
							}

							expectedClasses[k] = example;
							int split = example[0];
							Span<ushort> view = example.AsSpan(1, split);

							using (NewDisposeScope())
							{
								Tensor estimate = notchatgpt.Forward(simpleFullGPTDecoderUnit.EncodeOnly(example.AsSpan(1, example.Length - 2)), split - 1);
								estimate.MoveToOuterDisposeScope();
								actualTensors[k] = estimate;
							}
						}
						barrier2.SignalAndWait();
					}
				});
				thread.IsBackground = true;
				thread.Name = "Forward thread #" + i;
				thread.Start();
			}
			NLLLoss crossEntropyLoss = new NLLLoss(classweights, nn.Reduction.Mean);
			crossEntropyLoss.to(CUDA, ScalarType.Float64);
			
			


			Console.WriteLine("Start training...");

			for (int z = 0, savecooldown = 15; z < trainingBatches; ++z, --savecooldown)
			{
				Console.WriteLine("Forward pass batch #" + z);
				using var d2 = NewDisposeScope();
				adam.zero_grad();
				adam2.zero_grad();
				Tensor loss;
				barrier1.SignalAndWait();
				barrier2.SignalAndWait();

				int totallen = 0;
				for (int p = 0; p < trainingMicroBatchSize; ++p)
				{
					ushort[] arr = expectedClasses[p];
					totallen += arr.Length - (arr[0] + 1);
				}
				int[] ec2 = new int[totallen];
				for (int p = 0, pos = 0; p < trainingMicroBatchSize; ++p)
				{
					ushort[] arr = expectedClasses[p];
					int arrlen = arr.Length;

					for (int pos2 = arr[0] + 1; pos2 < arrlen; ++pos2, ++pos)
					{
						ec2[pos] = arr[pos2];
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

				bool defeat = totalloss2 < bestloss;
				if (savecooldown < (defeat ? 0 : -240))
				{
					if (defeat)
					{
						Console.WriteLine("Saving best policy...");
						bestloss = totalloss2;
					}
					else
					{
						Console.WriteLine("Saving policy...");
					}
					string savename = save + z;
					simpleFullGPTDecoderUnit.save(savename);
					savequeue.Enqueue(savename);
					if (savequeue.Count > 5)
					{
						File.Delete(savequeue.Dequeue());
					}
					savecooldown = 15;
				}
				Console.WriteLine("Optimizer step");

				learningRateScheduler.step();
				learningRateScheduler2.step();
				adam.step();
				adam2.step();
			}

		}
	}
}