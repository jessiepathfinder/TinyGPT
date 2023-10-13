using Newtonsoft.Json;
using System.Collections.Concurrent;
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
		//hyperparameters
		private const int latentTokenSize = 512;
		private const int maxContextSize = 2048;
		private const int trainingBatches = 200000;
		private const int trainingMicroBatchSize = 16;
		private const int attentionHeads = 8;
		const int secondTierAttentionDepth = 3;
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

			//2 magic token types
			tokenclasses += 3;
			Console.WriteLine("Loading ELI5 + WikiQA question answering dataset...");
			string[][]? questionanswering = JsonConvert.DeserializeObject<string[][]>(File.ReadAllText(datadir + "QuestionAnswering.json"));
			if (questionanswering is null)
			{
				Console.WriteLine("Null question answering dataset");
				return;
			}
			int wqlength = questionanswering.Length;

			Console.WriteLine("Starting dataset tokenizers...");

			ushort[][] tokenized = new ushort[wqlength][];
			int threads = Environment.ProcessorCount;
			int loadprogress = 0;
			Thread[] thrlist = new Thread[threads];
			int wqlen2 = wqlength;
			string progresstail = new StringBuilder("/").Append(wqlength).Append(" question-answer pairs").ToString();

			for (int z = 0; z < threads; ++z)
			{
				int az = z;
				Thread thread = new Thread(() =>
				{
					StringBuilder sb = new StringBuilder("Tokenized ");
					Span<ushort> encbuffer = stackalloc ushort[maxContextSize + 1];

					while (true)
					{
						int a = Interlocked.Increment(ref loadprogress);
						if (a > wqlength)
						{
							return;
						}
						a -= 1;
						string[] pair = questionanswering[a];

						Span<ushort> encbuffer2 = encbuffer.Slice(1, maxContextSize);
						int size1 = Transformer.Tokenize(dict, encbuffer2, pair[0], maxlen, 2);
						ushort encsize = (ushort)Math.Min(size1 + 1, maxContextSize - 1);
						if (size1 == maxContextSize)
						{
							goto flush;
						}
						encbuffer[size1++] = 0; //user-to-GPT context switch
						if (size1 == maxContextSize)
						{
							goto flush;
						}
						size1 += Transformer.Tokenize(dict, encbuffer2[size1..], pair[1], maxlen, 2);
						if (size1 == maxContextSize)
						{
							goto flush;
						}
						encbuffer[size1++] = 1; //GPT-to-user context switch

					flush:
						encbuffer[0] = encsize;
						tokenized[a] = encbuffer[..(size1 + 1)].ToArray();

						if ((a & 4095) == 4095)
						{
							Console.WriteLine(sb.Append(a).Append(progresstail).ToString());
							sb.Remove(10, sb.Length - 10);
						}

					}
				});
				thread.Name = "Data loader thread #" + z;
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
			GPTDecoderUnitV1 notchatgpt = new GPTDecoderUnitV1("TinyGPT", latentTokenSize, attentionHeads, tokenclasses, compressedViewSize, firstTierAttentionDepth, secondTierAttentionDepth, 1e-8);
			SimpleFullGPTDecoderUnit simpleFullGPTDecoderUnit = new SimpleFullGPTDecoderUnit(dictionaryItems, notchatgpt, "");

			simpleFullGPTDecoderUnit.to(CUDA, ScalarType.BFloat16);

			Adam adam = new Adam(notchatgpt.parameters(), lr: 1e-4, 0.95, weight_decay: 0.01, amsgrad: true, eps: 1e-8);
			SGD sgd = SGD(dictionaryItems.parameters(), 1e-4);
			LRScheduler learningRateScheduler = ExponentialLR(adam, 0.999, 0, true);
			LRScheduler learningRateScheduler2 = ExponentialLR(sgd, 0.999, 0, false);

			adam.to(CUDA);
			NLLLoss crossEntropyLoss = new NLLLoss(reduction: nn.Reduction.Mean);
			crossEntropyLoss.to(CUDA, ScalarType.BFloat16);
			notchatgpt.train(true);

			Console.WriteLine("Waiting for question answering dataset tokenization to complete...");
			foreach (Thread thr in thrlist)
			{
				thr.Join();
			}
			Console.WriteLine("Start multithreaded forward workers...");

			Barrier barrier1 = new Barrier(threads + 1);
			Barrier barrier2 = new Barrier(threads + 1);
			Tensor[] actualTensors = new Tensor[trainingMicroBatchSize];
			ushort[][] expectedClasses = new ushort[trainingMicroBatchSize][];
			for (int i = 1; i < trainingMicroBatchSize; ++i)
			{
				expectedClasses[i] = tokenized[RandomNumberGenerator.GetInt32(0, wqlen2)];
			}
			for (int i = 0; i < threads; ++i)
			{
				int z = i;
				Thread thread = new Thread(() => {
					while (true)
					{
						barrier1.SignalAndWait();
						for (int k = z; k < trainingMicroBatchSize; k += threads)
						{
							ushort[] example = expectedClasses[k];

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
			Console.WriteLine("Start training...");



			float bestloss = float.PositiveInfinity;
			Queue<string> savequeue = new Queue<string>();
			long[] shape1 = new long[] { 1, -1 };
			int maxgcgen = GC.MaxGeneration;

			for (int z = 0, savecooldown = 15; z < trainingBatches; ++z, --savecooldown)
			{
				Console.WriteLine("Forward pass batch #" + z);
				expectedClasses[z % trainingMicroBatchSize] = tokenized[RandomNumberGenerator.GetInt32(0, wqlen2)];
				using var d2 = NewDisposeScope();
				adam.zero_grad();
				sgd.zero_grad();
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
						lsp = cat2.log_softmax(0);
					}
					loss = crossEntropyLoss.forward(lsp, tensor(ec2).to(ScalarType.Int64, CUDA, true));

					loss.MoveToOuterDisposeScope();
				}
				for (int p = 0; p < trainingMicroBatchSize; ++p)
				{
					actualTensors[p].Dispose();
				}


				Console.WriteLine("Backpropagate batch #" + z);
				using (NewDisposeScope())
				{
					loss.backward();
				}

				float totalloss2 = loss.cpu().ToSingle();
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
				sgd.step();
			}

		}
	}
}