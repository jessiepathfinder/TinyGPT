using Newtonsoft.Json;
using System.Collections.Concurrent;
using System.Net.Http.Headers;
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
		private const int trainingBatches = 10000;
		private const int trainingMicroBatchSize = 16;
		private const int attentionHeads = 8;
		private const int feedForwardHiddenSize = 2048;
		private const int feedForwardDepth = 2;
		private const int prefinalhiddensize = 1024;




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
			if (dict is null) {
				Console.WriteLine("Null encoder dictionary");
				return;
			}
			int maxlen = 0;
			int tokenclasses = 0;
			foreach (KeyValuePair<string, ushort> keyValuePair in dict) {
				maxlen = Math.Max(maxlen, keyValuePair.Key.Length);
				tokenclasses = Math.Max(keyValuePair.Value, tokenclasses);
			}

			//2 magic token types
			tokenclasses += 2;
			Console.WriteLine("Loading ELI5 + WikiQA question answering dataset...");
			string[][]? questionanswering = JsonConvert.DeserializeObject<string[][]>(File.ReadAllText(datadir + "QuestionAnswering.json"));
			if (questionanswering is null) {
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

			for (int z = 0; z < threads; ++z) {
				int az = z;
				Thread thread = new Thread(() =>
				{
					StringBuilder sb = new StringBuilder("Tokenized ");
					Span<ushort> encbuffer = stackalloc ushort[maxContextSize + 1];

					while (true)
					{
						int a = Interlocked.Increment(ref loadprogress);
						if (a > wqlength) {
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
			GPTDecoderUnitV1 notchatgpt = new GPTDecoderUnitV1("TinyGPT", latentTokenSize, attentionHeads, feedForwardDepth, feedForwardHiddenSize, tokenclasses, prefinalhiddensize);
			SimpleFullGPTDecoderUnit simpleFullGPTDecoderUnit = new SimpleFullGPTDecoderUnit(dictionaryItems, notchatgpt, "");

			simpleFullGPTDecoderUnit.to(CUDA, ScalarType.Float32);

			Adam adam = new Adam(notchatgpt.parameters(), lr: 0.0005, 0.95, amsgrad: true);
			SGD sgd = SGD(dictionaryItems.parameters(), 0.001);
			LRScheduler learningRateScheduler = ExponentialLR(adam, 0.999, 1, true);

			adam.to(CUDA);
			CrossEntropyLoss crossEntropyLoss = new CrossEntropyLoss(reduction: nn.Reduction.Mean);
			crossEntropyLoss.to(CUDA, ScalarType.Float32);
			notchatgpt.train(true);

			Console.WriteLine("Waiting for question answering dataset tokenization to complete...");
			foreach (Thread thr in thrlist)
			{
				thr.Join();
			}

			Console.WriteLine("Start training...");
			float bestloss = float.PositiveInfinity;
			Queue<string> savequeue = new Queue<string>();
			long[] shape1 = new long[] {1, -1};
			int maxgcgen = GC.MaxGeneration;
			Tensor[] actualTensors = new Tensor[trainingMicroBatchSize];
			for (int z = 0, savecooldown = 15; z < trainingBatches; ++z, --savecooldown)
			{
				
				
				Console.WriteLine("Forward pass batch #" + z);
				using var d2 = NewDisposeScope();
				adam.zero_grad();
				sgd.zero_grad();
				Tensor loss;
				
				
				using(NewDisposeScope()){
					List<long> expectedTensorList = new List<long>();
					for (int k = 0; k < trainingMicroBatchSize; ++k)
					{
						ushort[] example = tokenized[RandomNumberGenerator.GetInt32(wqlen2)];

						int split = example[0];
						ushort backup = example[split + 1];
						Span<ushort> view = example.AsSpan(1, split);

						using (NewDisposeScope())
						{
							Tensor estimate = notchatgpt.Forward(simpleFullGPTDecoderUnit.EncodeOnly(example.AsSpan(1)), split);
							estimate.MoveToOuterDisposeScope();
							actualTensors[k] = estimate;
						}
						++split;
						int len = example.Length;
						expectedTensorList.Capacity += (len - split);
						while (split < len)
						{
							expectedTensorList.Add(example[split++]);
						}

					}
					Console.WriteLine("Compute loss batch #" + z);
					loss = crossEntropyLoss.forward(cat(actualTensors, 0), tensor(expectedTensorList).to(CUDA));
					loss.MoveToOuterDisposeScope();
				}

				float totalloss2 = loss.cpu().ToSingle();
				Console.WriteLine("Batch loss: " + totalloss2);

				if (totalloss2 < bestloss & savecooldown < 0)
				{
					Console.WriteLine("Saving best policy...");
					bestloss = totalloss2;
					string savename = save + z;
					notchatgpt.save(savename);
					savequeue.Enqueue(savename);
					if (savequeue.Count > 5)
					{
						File.Delete(savequeue.Dequeue());
					}
					savecooldown = 15;
				}
				Console.WriteLine("Backpropagate batch #" + z);
				loss.backward();
				learningRateScheduler.step();
				adam.step();
				sgd.step();
			}

		}
	}
}