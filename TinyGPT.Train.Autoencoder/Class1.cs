using System.Collections.Concurrent;
using System.Formats.Tar;
using System.IO.Compression;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn.functional;

namespace TinyGPT.Train
{
	static class Program{
		private readonly struct RecyclerRecord{
			public readonly float[] array;
			public readonly int erase;

			public RecyclerRecord(float[] array, int erase)
			{
				this.array = array;
				this.erase = erase;
			}
		}
		private static void Main(string[] args){
			string datadir = args[0];
			string encodersave = args[1];
			string decodersave = args[2];
			if (!datadir.EndsWith(Path.DirectorySeparatorChar)){
				datadir += Path.DirectorySeparatorChar;
			}
			Console.WriteLine("Loading wordlist...");
			IReadOnlyDictionary<string, float> wordlist = Tokenizer.WordListToDictionary(File.ReadAllLines(datadir + "common_english_words.txt"));
			Console.WriteLine("Loading unlableled wikipedia data...");
			string[] strings = new string[12217];
			using (TarReader tarReader = new TarReader(new GZipStream(new BufferedStream(new FileStream(datadir + "stripped_wikipedia.tar.gz", FileMode.Open, FileAccess.Read, FileShare.Read, 16777216, FileOptions.SequentialScan), 16777216), CompressionMode.Decompress, false), false)) {
				int z = 0;
				while(true){
					TarEntry? tarEntry = tarReader.GetNextEntry(false);
					if(tarEntry is null){
						break;
					}
					Stream? datastream = tarEntry.DataStream;
					if (datastream is null) {
						continue;
					}
					using(StreamReader streamReader = new StreamReader(datastream, Encoding.UTF8, false, -1, true)){
						strings[z++] = streamReader.ReadToEnd();
					}
					if((z & 1023) == 0){
						Console.WriteLine("Loading unlableled wikipedia data: " + z + "/12217 articles loaded");
					}
				}
			}

			Console.WriteLine("starting data loader threads...");

			using SemaphoreSlim semaphoreSlim1 = new SemaphoreSlim(16, 4096);
			using SemaphoreSlim semaphoreSlim2 = new SemaphoreSlim(0, 1048576);

			ConcurrentBag<RecyclerRecord> traindata = new ConcurrentBag<RecyclerRecord>();
			ConcurrentBag<RecyclerRecord> recycler = new ConcurrentBag<RecyclerRecord>();
			int threads = Environment.ProcessorCount;
			for(int i = 0; i < threads; ++i){
				int threadid = i;
				Thread thread = new Thread(() => {
					while(true){
						semaphoreSlim1.Wait();
						for(int q = 0; q < 256; ++q){
							string str = strings[RandomNumberGenerator.GetInt32(0, 12217)];
							float[] data;
							int erase;
							if (recycler.TryTake(out RecyclerRecord recyclerRecord))
							{
								data = recyclerRecord.array;
								erase = recyclerRecord.erase;
							}
							else
							{
								data = new float[1024];
								erase = 1024;
							}
							int offset = Tokenizer.Tokenize(str.AsSpan()[RandomNumberGenerator.GetInt32(0, str.Length)..], 31, data, wordlist);
							for (int p = Math.Max(erase, offset); p < 1024; ++p)
							{
								data[p] = 0f;
							}
							traindata.Add(new RecyclerRecord(data, offset));
							semaphoreSlim2.Release();
						}
					}
				});
				thread.IsBackground = true;
				thread.Name = "TinyGPT Autoencoder training data fetcher thread #" + i;
				thread.Start();
			}

			Console.WriteLine("Initializing Torch...");
			InitializeDeviceType(DeviceType.CUDA);
			Encoderv1 encoder = new Encoderv1("");
			encoder.to(CUDA);
			Decoderv1 decoder = new Decoderv1("");
			decoder.to(CUDA);
			VAECombinedNetwork model = new VAECombinedNetwork(encoder, decoder, "");
			model.to(CUDA);
			Adam optimizer = optim.Adam(model.parameters(), amsgrad: true);
			//optim.lr_scheduler.StepLR(optimizer, 1);
			optimizer.to(CUDA);
			model.train();


			Console.WriteLine("Start training!");
			

			for (int i = 0; i < 4096; ++i){
				if(i< 4080)
				{
					semaphoreSlim1.Release();
				}
				

				using (var d = NewDisposeScope())
				{
					optimizer.zero_grad();


					Tensor kldivergence = null;
					Tensor reconstruction = null;
					Tensor sum = null;
					for (int p = 0; p < 256; ++p)
					{
						semaphoreSlim2.Wait();
						traindata.TryTake(out RecyclerRecord data);
						Tensor input = data.array;
						input = input.to(CUDA);
						recycler.Add(data);

						(Tensor x, Tensor y, Tensor z) = model.forward(input);
						Tensor zsquare = z.square();
						
						Tensor a = mse_loss(x, input);
						Tensor b = torch.sum(y.square().add(zsquare).sub(zsquare.log())) / 512.0f;
						Tensor c = a.add(b);
						c.backward();
						if (p == 0){
							reconstruction = a;
							kldivergence = b;
							sum = c;
						} else{
							reconstruction = reconstruction.add(a);
							kldivergence = kldivergence.add(b);
							sum = sum.add(c);
						}
					}


					

					optimizer.step();
					Console.WriteLine();
					Console.WriteLine("Epouch " + i);
					Console.WriteLine("Reconstruction Cost: " + reconstruction.to(CPU).ToSingle());
					Console.WriteLine("KL Divergence Cost: " + kldivergence.to(CPU).ToSingle());
					Console.WriteLine("Total Cost: " + sum.to(CPU).ToSingle());
					Console.WriteLine("Array recycler count: " + recycler.Count);
					Console.WriteLine("Input buffer count: " + traindata.Count);
				}
				
			}
			encoder.save(encodersave);
			decoder.save(decodersave);
		}

	}
}