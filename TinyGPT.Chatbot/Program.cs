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


		//hyperparameters
		private const int latentTokenSize = 512;
		private const int maxContextSize = 1024;
		private const int trainingBatches = 200000;
		private const int trainingMicroBatchSize = 16;
		private const int attentionHeads = 12;
		private const int firstTierAttentionDepth = 8;
		private const int compressedPreFinalSize = 2048;



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
			string progresstail = new StringBuilder("/").Append(wqlen2).Append(" question-answer pairs").ToString();

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


						int size1 = Transformer.Tokenize(dict, encbuffer2, pair[0], maxlen, 2);
						ushort encsize = (ushort)Math.Min(size1 + 1, maxContextSize - 1);
						int encsize2 = size1;
						if (size1 == maxContextSize)
						{
							continue;
						}
						encbuffer[size1++] = 0; //user-to-GPT context switch
						if (size1 == maxContextSize)
						{
							continue;
						}
						int ctd = Transformer.Tokenize(dict, encbuffer2[size1..], pair[1], maxlen, 2);
						if (ctd == 0)
						{
							continue;
						}
						size1 += ctd;
						if (size1 < maxContextSize)
						{
							encbuffer[size1++] = 1; //GPT-to-user context switch
													//Memory regularization
							int copysize = Math.Min(maxContextSize - size1, encsize2);
							if (copysize > 0)
							{
								encbuffer[..copysize].CopyTo(encbuffer[size1..]);
								size1 += copysize;

							}
						}

						++size1;
						encbuffer[0] = encsize;
						tokenized[a] = encbuffer[..(size1)].ToArray();

						if ((a & 4095) == 4095)
						{
							Console.WriteLine(sb.Append(a).Append(progresstail).ToString());
							sb.Remove(10, sb.Length - 10);
						}

					}

					encbuffer2 = encbuffer.Slice(0, maxContextSize - 1);

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
			GPTDecoderUnitV1 notchatgpt = new GPTDecoderUnitV1("TinyGPT", latentTokenSize, attentionHeads, tokenclasses, firstTierAttentionDepth, compressedPreFinalSize, 1e-9);
			SimpleFullGPTDecoderUnit simpleFullGPTDecoderUnit = new SimpleFullGPTDecoderUnit(dictionaryItems, notchatgpt, "");

			simpleFullGPTDecoderUnit.to(CUDA, ScalarType.BFloat16);
			Adam adam2 = new Adam(dictionaryItems.parameters(), lr: 1e-5, amsgrad: true, eps: 1e-9);
			Adam adam = new Adam(notchatgpt.parameters(), lr: 1e-5, amsgrad: true, weight_decay: 1e-3, eps: 1e-9);
			LRScheduler learningRateScheduler = ExponentialLR(adam, 0.9998, 0, true);
			LRScheduler learningRateScheduler2 = ExponentialLR(adam2, 0.9998, 0, false);
			NLLLoss crossEntropyLoss = new NLLLoss(reduction: nn.Reduction.Mean);
			crossEntropyLoss.to(CUDA, ScalarType.Float64);
			adam.to(CUDA);
			adam2.to(CUDA);

			simpleFullGPTDecoderUnit.train(true);






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

			Console.WriteLine("Optimizing memory usage...");


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
							ushort[]? example = null;
							while (example is null)
							{
								example = tokenized[RandomNumberGenerator.GetInt32(0, wqlen2)];
							}

							expectedClasses[k] = example;
							int split = example[0];
							Span<ushort> view = example.AsSpan(1, split);

							using (NewDisposeScope())
							{
								Tensor estimate = notchatgpt.Forward(simpleFullGPTDecoderUnit.EncodeOnly(example.AsSpan(1, example.Length - 2)), split);
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
					totallen += arr.Length - (arr[0] + 2);
				}
				int[] ec2 = new int[totallen];
				for (int p = 0, pos = 0; p < trainingMicroBatchSize; ++p)
				{
					ushort[] arr = expectedClasses[p];
					int arrlen = arr.Length;

					for (int pos2 = arr[0] + 2; pos2 < arrlen; ++pos2)
					{
						ec2[pos++] = arr[pos2];
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
				GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, true, false);
				GC.WaitForPendingFinalizers();




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