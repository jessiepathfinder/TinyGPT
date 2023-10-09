using Newtonsoft.Json;
using System.Runtime.Serialization.Formatters;
using TinyGPT.Core;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using Transformer = TinyGPT.Core.Transformer;

namespace TinyGPT.Chatbot
{
	internal class Program
	{
		static void Main(string[] args)
		{
			string datadir = args[0];
			string model = args[1];
			if (!datadir.EndsWith(Path.DirectorySeparatorChar))
			{
				datadir += Path.DirectorySeparatorChar;
			}
			Console.WriteLine("Loading dictioanry...");
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
			tokenclasses += 2;
			string[] decode = new string[tokenclasses + 2];
			int maxtokensize = 0;
			foreach(KeyValuePair<string, ushort> keyValuePair in dict){
				string key = keyValuePair.Key;
				decode[keyValuePair.Value + 3] = key;
				maxtokensize = Math.Max(maxtokensize, key.Length);
			}
			Console.WriteLine("Loading model...");
			bool usecuda = cuda_is_available();
			if( usecuda ) {
				InitializeDeviceType(DeviceType.CUDA);
			} else{
				Console.WriteLine("WARNING: Your computer does not have a NVIDIA CUDA Graphics Card, or lacks the needed graphics drivers (I will be very slow)");
			}
			FullGPTDecoderUnit themodel;
			int maxcontext;
			switch(model){
				case "nano-v1":
					{
						const int latentTokenSize = 512;
						maxcontext = 2048;
						const int trainingBatches = 100000;
						const int trainingMicroBatchSize = 16;
						const int attentionHeads = 8;
						const int feedForwardHiddenSize = 2048;
						const int feedForwardDepth = 3;
						const int compressedViewSize = 1024;
						const int processorHiddenSize = 1024;
						const int processorDepth = 3;

						ModuleList<BERTDictionaryItem> dictionaryItems = new ModuleList<BERTDictionaryItem>();
						for (int i = 0; i < tokenclasses; ++i)
						{
							dictionaryItems.Add(new BERTDictionaryItem("", latentTokenSize));
						}
						GPTDecoderUnitV1 notchatgpt = new GPTDecoderUnitV1("TinyGPT", latentTokenSize, attentionHeads, feedForwardDepth, feedForwardHiddenSize, tokenclasses, compressedViewSize, processorDepth, processorHiddenSize);
						themodel = new SimpleFullGPTDecoderUnit(dictionaryItems, notchatgpt, "");

					}
					break;
				default:
					Console.WriteLine("unknown model!");
					return;
			}
			themodel.load(datadir + model + ".model");
			themodel.eval();
			if (usecuda)
			{
				themodel.to(CUDA, ScalarType.Float32);
			}

			Span<ushort> buffer = stackalloc ushort[maxcontext];
			int maxinsize = maxcontext - 2;
			Console.WriteLine("Ready!");
			while(true){
				Console.Write("User: ");
				string? input = Console.ReadLine();
				if(input is null){
					continue;
				}
				Console.Write("TinyGPT: ");
				int intokens = Transformer.Tokenize(dict, buffer, input, maxtokensize, 2);
				if(intokens > maxinsize){
					Console.WriteLine("too big!");
					continue;
				}
				buffer[intokens] = 0; //[STARTGPT]
				for(int i = intokens + 1; i < maxcontext; ++i){
					float best = -1;
					int bestindex = 1;
					Tensor tensor;
					using (var ds = NewDisposeScope()){
						tensor = themodel.Forward(buffer.Slice(0, i)).cpu();
						tensor.MoveToOuterDisposeScope();
					}
					using(tensor){
						for (int z = 0; z < tokenclasses; ++z)
						{
							float my = tensor[z].ToScalar().ToSingle();
							if (my > best)
							{
								best = my;
								bestindex = z;
							}
						}
					}

					if (bestindex == 1){
						break;
					}
					string? str = decode[bestindex];
					buffer[i] = (ushort)bestindex;
					if (str is null){
						continue;
					}
					Console.Write(str);
				}
				Console.WriteLine();
				Console.WriteLine("==================================================");
				Console.WriteLine();

			}
		}
	}
}