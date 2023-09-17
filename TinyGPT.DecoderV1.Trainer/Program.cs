using Newtonsoft.Json;
using System.Collections.Concurrent;
using System.Security.Cryptography;
using System.Text;
using System.Xml.Schema;
using TinyGPT.Core;
using TorchSharp;
using TorchSharp.Modules;
using static TinyGPT.Core.GPTDecoderV1;
using static TorchSharp.torch;
using static TorchSharp.torch.optim;
using Transformer = TinyGPT.Core.Transformer;

namespace TinyGPT.DecoderV1.Trainer
{
	internal static class Program
	{
		//hyperparameters
		private const int latentTokenSize = 256;
		private const int unmaskedAttentionHeadsCount = 32;
		private const int maskedAttentionHeadsCount = 32;
		private const int maxContextSize = 1024;
		private const int fullDataTransitionIndex = 3000;
		private const int trainingBatches = 5000;
		private const int trainingBatchSize = 256;
		private const int transformerAttentionHeads = 16;
		private const int transformerDepth = 2;
		private const int predictorDepth = 5;

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
			if(dict is null){
				Console.WriteLine("Null encoder dictionary");
				return;
			}
			int maxlen = 0;
			int tokenclasses = 0;
			foreach(KeyValuePair<string, ushort> keyValuePair in dict){
				maxlen = Math.Max(maxlen, keyValuePair.Key.Length);
				tokenclasses = Math.Max(keyValuePair.Value, tokenclasses);
			}

			//3 magic token types
			tokenclasses += 3;
			Console.WriteLine("Loading ELI5 + WikiQA question answering dataset...");
			string[][]? questionanswering = JsonConvert.DeserializeObject<string[][]>(File.ReadAllText(datadir + "QuestionAnswering.json"));
			if(questionanswering is null){
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

			for (int z = 0; z < threads; ++z){
				int az = z;
				Thread thread = new Thread(() =>
				{
					StringBuilder sb = new StringBuilder("Tokenized ");
					Span<ushort> encbuffer = stackalloc ushort[maxContextSize + 1];
					
					while (true)
					{
						int a = Interlocked.Increment(ref loadprogress);
						if(a >= wqlength){
							return;
						}
						string[] pair = questionanswering[a];
						
						Span<ushort> encbuffer2 = encbuffer.Slice(1, maxContextSize);
						int size1 = Transformer.Tokenize(dict, encbuffer2, pair[0], maxlen);
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
						size1 += Transformer.Tokenize(dict, encbuffer2[size1..], pair[1], maxlen);
						if (size1 == maxContextSize)
						{
							goto flush;
						}
						encbuffer[size1++] = 1; //GPT-to-user context switch

					flush:
						encbuffer[0] = encsize;
						tokenized[a] = encbuffer[..size1].ToArray();

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
			for(int i = 0; i < tokenclasses; ++i){
				dictionaryItems.Add(new BERTDictionaryItem("", latentTokenSize));
			}
			
			FullGPTDecoderUnitV1 notchatgpt = new FullGPTDecoderUnitV1("TinyGPT", dictionaryItems, new GPTDecoderV1(transformerDepth, transformerAttentionHeads, predictorDepth, latentTokenSize, tokenclasses, ""));
			notchatgpt.to(CUDA, ScalarType.Float32);

			Adam adam = new Adam(notchatgpt.parameters(), amsgrad: true);
			adam.to(CUDA);
			notchatgpt.train(true);

			Console.WriteLine("Waiting for question answering dataset tokenization to complete...");
			foreach (Thread thr in thrlist)
			{
				thr.Join();
			}

			Console.WriteLine("Start training...");
			for (int i = 0; i < trainingBatches; ++i)
			{
				
				Console.WriteLine("Start training batch #" + i);
				using (var d = NewDisposeScope())
				{
					adam.zero_grad();
					float totalloss = 0;
					for(int z = 0; z < trainingBatchSize; ++z){
						
						ushort[] example = tokenized[RandomNumberGenerator.GetInt32(wqlen2)];

						int split = RandomNumberGenerator.GetInt32(example[0], example.Length - 1);
						ushort backup = example[split];
						example[split] = 3; //MASK token
						Span<ushort> view = example.AsSpan(1, split + 1);

						Tensor prob = notchatgpt.forward(view);

						Tensor loss = Misc.ComputeSoftmaxLoss2(prob, backup);
						example[split] = backup;
						loss.backward();
						totalloss += (float)loss.cpu();
					}
					Console.WriteLine("Batch total loss: " + totalloss);
					
					adam.step();
				}
			}
			notchatgpt.Save(save);

		}
	}
}