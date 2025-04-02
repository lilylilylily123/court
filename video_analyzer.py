import os
import cv2
import openai
import base64
import glob
import subprocess
import asyncio
import aiohttp
import logging
from dotenv import load_dotenv
from PIL import Image
import imagehash
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load API Key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OpenAI API key not found. Check your .env file.")
openai.api_key = api_key

# ==============================
# STEP 1: EXTRACT KEY FRAMES
# ==============================
def extract_key_frames(video_path, output_folder="frames", threshold=5):
    logging.info("Starting frame extraction process...")
    # Check if video file exists
    if not os.path.exists(video_path):
        logging.error(f"Video file not found: {video_path}")
        raise FileNotFoundError(f"Video file not found: {video_path}")

    logging.info(f"Creating output folder: {output_folder}")
    os.makedirs(output_folder, exist_ok=True)
    cap = None
    last_hash = None
    frame_count = 0
    saved_count = 0
    start_time = time.time()
    progress_interval = 100

    try:
        logging.info(f"Opening video file: {video_path}")
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            logging.error(f"Failed to open video file: {video_path}")
            raise ValueError(f"Failed to open video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        logging.info(f"Video has {total_frames} frames, {fps:.2f} fps, duration: {duration:.2f} seconds")

        while cap.isOpened():
            try:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                if frame_count % progress_interval == 0:
                    elapsed = time.time() - start_time
                    percent_done = (frame_count / total_frames * 100) if total_frames > 0 else 0
                    logging.debug(f"Processed {frame_count}/{total_frames} frames ({percent_done:.1f}%) - Time elapsed: {elapsed:.1f}s")

                if frame_count % 10 != 0:
                    continue

                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((256, 256))
                current_hash = imagehash.phash(img)

                if last_hash is None or abs(current_hash - last_hash) > threshold:
                    frame_path = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
                    img.save(frame_path)
                    logging.debug(f"Saved frame {saved_count} at position {frame_count} to {frame_path}")
                    saved_count += 1
                    last_hash = current_hash
            except KeyboardInterrupt:
                print("\nKeyboard interrupt detected during frame extraction. Cleaning up...")
                break
    except Exception as e:
        logging.error(f"Error during frame extraction: {str(e)}")
        raise
    finally:
        if cap is not None:
            cap.release()
    elapsed_time = time.time() - start_time
    logging.info(f"Frame extraction complete. Extracted {saved_count} key frames from {frame_count} processed frames.")
    logging.info(f"Frame extraction took {elapsed_time:.2f} seconds ({frame_count/elapsed_time:.1f} frames/sec)")
    print(f"Extracted {saved_count} key frames.")

# ==============================
# STEP 2: EXTRACT AUDIO
# ==============================
def extract_audio(video_path, output_audio="audio.wav"):
    logging.info(f"Starting audio extraction from {video_path}...")
    # Check if video file exists
    if not os.path.exists(video_path):
        logging.error(f"Video file not found: {video_path}")
        raise FileNotFoundError(f"Video file not found: {video_path}")

    try:
        start_time = time.time()
        logging.info(f"Running ffmpeg to extract audio to {output_audio}")
        command = f"ffmpeg -i {video_path} -vn -acodec pcm_s16le -ar 16000 -ac 1 {output_audio} -y"

        # Execute the command
        result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if result.returncode != 0:
            error_message = result.stderr.decode('utf-8')
            logging.error(f"Failed to extract audio: {error_message}")
            raise RuntimeError(f"Failed to extract audio: {error_message}")

    except Exception as e:
        logging.error(f"Error extracting audio: {str(e)}")
        raise

    elapsed_time = time.time() - start_time
    logging.info(f"Audio extraction complete. Took {elapsed_time:.2f} seconds.")

# ==============================
# STEP 3: TRANSCRIBE AUDIO
# ==============================
def transcribe_audio(audio_path, output_transcript="transcript.txt"):
    logging.info(f"Starting audio transcription of {audio_path}...")

    # Verify audio file exists
    if not os.path.exists(audio_path):
        logging.error(f"Audio file not found: {audio_path}")
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    start_time = time.time()
    whisper_exec = "/Users/Lily/projects/court/whisper.cpp/build/bin/whisper-cli"
    model_path = "/Users/Lily/projects/court/whisper.cpp/models/ggml-base.bin"

    # Verify whisper executable exists
    if not os.path.exists(whisper_exec):
        error_msg = f"Whisper executable not found at: {whisper_exec}"
        logging.error(error_msg)
        raise FileNotFoundError(error_msg)

    # Verify model file exists
    if not os.path.exists(model_path):
        error_msg = f"Whisper model not found at: {model_path}"
        logging.error(error_msg)
        raise FileNotFoundError(error_msg)

    # Expected output filename from whisper - whisper adds .txt to the audio filename
    expected_output = f"{audio_path}.txt"

    logging.info(f"Running Whisper transcription with model: {os.path.basename(model_path)}")
    command = f"{whisper_exec} -f {audio_path} -m {model_path} --output-txt --no-gpu"

    try:
        logging.info(f"Executing command: {command}")

        # Execute command and capture output
        result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Log command output for debugging
        if result.stdout:
            logging.debug(f"Command stdout: {result.stdout.strip()}")

        if result.returncode != 0:
            error_message = result.stderr
            logging.error(f"Transcription failed: {error_message}")
            raise RuntimeError(f"Transcription failed: {error_message}")

        # Check for the output file with correct name
        if os.path.exists(expected_output):
            os.rename(expected_output, output_transcript)
            logging.info(f"Transcription saved to {output_transcript}")
        else:
            logging.error(f"Transcription output file '{expected_output}' not found")
            raise FileNotFoundError(f"Transcription output file '{expected_output}' not found")
    except Exception as e:
        logging.error(f"Error during transcription: {str(e)}")
        raise

    elapsed_time = time.time() - start_time
    logging.info(f"Transcription complete. Took {elapsed_time:.2f} seconds.")
    print(f"✅ Audio transcription completed in {elapsed_time:.2f} seconds")

# ==============================
# STEP 4: ANALYZE TEXT
# ==============================
def analyze_text(transcript_path):
    logging.info(f"Starting text analysis of transcript: {transcript_path}")

    if not os.path.exists(transcript_path):
        logging.error(f"Transcript file not found: {transcript_path}")
        raise FileNotFoundError(f"Transcript file not found: {transcript_path}")

    start_time = time.time()
    try:
        with open(transcript_path, "r") as file:
            transcript = file.read()

        logging.info(f"Sending transcript to OpenAI for analysis (length: {len(transcript)} chars)")
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=[{"role": "user", "content": f"Summarize this transcript in 3 sentences:\n{transcript}"}], max_tokens=200
        )

        summary = response['choices'][0]['message']['content']
        elapsed_time = time.time() - start_time
        logging.info(f"Text analysis complete. Took {elapsed_time:.2f} seconds.")
        return summary
    except Exception as e:
        logging.error(f"Error analyzing transcript: {str(e)}")
        raise

# ==============================
# STEP 5: ASYNC IMAGE PROCESSING WITH BATCHING AND RATE LIMIT HANDLING
# ==============================
def encode_image(image_path):
    with open(image_path, "rb") as img:
        return base64.b64encode(img.read()).decode('utf-8')

async def analyze_image(session, image_path, model="gpt-4o-mini"):
    image_data = encode_image(image_path)
    messages = [
        {"role": "user", "content": "Summarize this image in 10 words or fewer, stating only visible facts. Do not include any personal opinions or interpretations."},
        {"role": "user", "content": f"data:image/jpeg;base64,{image_data}"}
    ]
    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        try:
            logging.info(f"Analyzing image {image_path} with model {model}")
            response = await session.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {openai.api_key}"},
                json={"model": model, "messages": messages, "max_tokens": 50},
            )

            # Handle rate limiting
            if response.status == 429:
                retry_after = int(response.headers.get('Retry-After', 30))  # Default to 30 seconds if no header
                logging.warning(f"Rate limit hit, retrying after {retry_after} seconds...")
                await asyncio.sleep(retry_after)
                continue

            # Check other errors
            if response.status != 200:
                error_text = await response.text()
                logging.error(f"API error: HTTP {response.status} - {error_text}")
                return f"Error analyzing image: HTTP {response.status}"

            # Parse the response
            result = await response.json()
            if "choices" in result and result["choices"]:
                choice = result["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    return choice["message"]["content"]
                else:
                    logging.error(f"Unexpected response structure: {choice}")
                    return "Error: Unexpected API response structure"
            else:
                logging.error(f"Unknown API response format: {result}")
                return "Error: Could not parse API response"

        except aiohttp.ClientError as e:
            logging.error(f"HTTP client error: {str(e)}")
            retry_count += 1
            if retry_count < max_retries:
                wait_time = 2 ** retry_count
                logging.warning(f"Network error, retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
            else:
                return f"Network error analyzing image: {str(e)}"
        except Exception as e:
            logging.error(f"Unexpected error analyzing image {image_path}: {str(e)}")
            return f"Error analyzing image: {str(e)}"

    return "Failed to analyze image after multiple retries"

async def process_all_frames():
    logging.info("Starting image analysis phase...")
    print("\n🔍 Starting image analysis phase...")

    # Verify frames exist
    frames = sorted(glob.glob("frames/*.jpg"))
    if not frames:
        error_msg = "No frames found in 'frames' directory. Cannot proceed with image analysis."
        logging.error(error_msg)
        print(f"❌ ERROR: {error_msg}")
        raise FileNotFoundError("No frames found for image analysis")

    logging.info(f"Found {len(frames)} frames to analyze")
    print(f"📊 Found {len(frames)} frames to analyze in batches")

    start_time = time.time()
    results = []
    errors = []
    batch_stats = {"success": 0, "failed": 0, "skipped": 0}

    async with aiohttp.ClientSession() as session:
        batch_size = 5  # Adjust batch size as necessary
        total_batches = (len(frames) - 1) // batch_size + 1
        print(f"⏳ Processing {len(frames)} frames in {total_batches} batches...")

        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(frames))
            batch = frames[start_idx:end_idx]

            batch_msg = f"Processing batch {batch_num+1}/{total_batches} (frames {start_idx+1}-{end_idx})"
            logging.info(batch_msg)
            print(f"\n🔄 BATCH {batch_num+1}/{total_batches}: Processing frames {start_idx+1}-{end_idx}...")

            # Stats for this batch
            batch_errors = 0
            batch_skipped = 0

            # Create tasks for the current batch
            tasks = []
            frame_paths = []  # Store paths to match with results

            for i, frame in enumerate(batch):
                # Verify frame exists
                if not os.path.exists(frame):
                    logging.warning(f"Frame file not found: {frame} - skipping")
                    print(f"⚠️  Frame not found: {os.path.basename(frame)} - skipping")
                    batch_skipped += 1
                    batch_stats["skipped"] += 1
                    continue

                # Alternate between models for different frames
                model_choice = "gpt-4o-mini" if (start_idx + i) % 2 == 0 else "gpt-3.5-turbo"
                logging.debug(f"Adding frame {frame} to queue (using model: {model_choice})")
                frame_paths.append(frame)
                tasks.append(analyze_image(session, frame, model=model_choice))

            # Process this batch and collect results
            if tasks:
                try:
                    batch_results = await asyncio.gather(*tasks, return_exceptions=True)

                    # Process each result
                    for j, result in enumerate(batch_results):
                        if isinstance(result, Exception):
                            # Handle exception
                            error_msg = f"Error processing frame {frame_paths[j]}: {str(result)}"
                            logging.error(error_msg)
                            errors.append(error_msg)
                            print(f"❌ Failed: {os.path.basename(frame_paths[j])}")
                            batch_errors += 1
                            batch_stats["failed"] += 1
                        elif isinstance(result, str) and result.startswith("Error"):
                            # Handle error string from the analyze_image function
                            logging.error(result)
                            errors.append(result)
                            print(f"❌ Failed: {os.path.basename(frame_paths[j])}")
                            batch_errors += 1
                            batch_stats["failed"] += 1
                        else:
                            # Success - add to results
                            results.append(result)
                            print(f"✅ Analyzed: {os.path.basename(frame_paths[j])}")

                    # Batch summary
                    success_count = len(batch_results) - batch_errors
                    batch_stats["success"] += success_count

                    progress_pct = (len(results) / len(frames)) * 100
                    elapsed = time.time() - start_time

                    batch_status = "✅ Success" if batch_errors == 0 else "⚠️  Partial success" if success_count > 0 else "❌ Failed"
                    logging.info(f"Batch {batch_num+1} complete. Status: {batch_status}")
                    print(f"\n{batch_status}: Batch {batch_num+1}/{total_batches} - "
                          f"{success_count} succeeded, {batch_errors} failed, {batch_skipped} skipped")
                    print(f"📈 Overall progress: {len(results)}/{len(frames)} frames processed ({progress_pct:.1f}%) - "
                          f"Time elapsed: {elapsed:.1f}s")

                except Exception as e:
                    error_msg = f"Batch {batch_num+1} failed completely: {str(e)}"
                    logging.error(error_msg)
                    errors.append(error_msg)
                    print(f"\n❌ BATCH FAILED: Batch {batch_num+1}/{total_batches} - {str(e)}")
                    batch_stats["failed"] += len(tasks)
            else:
                logging.warning(f"Batch {batch_num+1} had no valid frames to process")
                print(f"⚠️  Batch {batch_num+1}/{total_batches} had no valid frames to process")

            # Add delay between batches to avoid rate limiting (if not the last batch)
            if batch_num < total_batches - 1:
                logging.info(f"Waiting 5 seconds before next batch to avoid rate limiting...")
                print(f"⏳ Waiting 5 seconds before next batch to avoid rate limiting...")
                await asyncio.sleep(5)  # Adjust as necessary

        elapsed_time = time.time() - start_time

        # Final statistics
        logging.info(f"Image analysis complete. Analyzed {len(results)} frames in {elapsed_time:.2f} seconds.")
        print(f"\n=== IMAGE ANALYSIS SUMMARY ===")
        print(f"✅ Success: {batch_stats['success']} frames")
        print(f"❌ Failed: {batch_stats['failed']} frames")
        print(f"⚠️  Skipped: {batch_stats['skipped']} frames")
        print(f"⏱️  Time elapsed: {elapsed_time:.2f} seconds ({len(frames)/elapsed_time:.2f} frames/sec)")

        # Report errors if any
        if errors:
            print(f"\n❌ ERROR SUMMARY ({len(errors)} issues detected):")
            for i, error in enumerate(errors[:5]):  # Show only first 5 errors to avoid flooding
                print(f"  {i+1}. {error[:100]}..." if len(error) > 100 else f"  {i+1}. {error}")

            if len(errors) > 5:
                print(f"  ... and {len(errors) - 5} more errors")

            logging.error(f"Image analysis completed with {len(errors)} errors")
        else:
            print(f"\n✅ All frames analyzed successfully!")

        return results

def full_analysis(text_summary, image_summaries):
    """
    Combines text and image analysis into a comprehensive report.

    Args:
        text_summary: Summary of transcribed audio
        image_summaries: List of summaries for key frames

    Returns:
        Comprehensive analysis text
    """
    logging.info("Starting comprehensive analysis combining text and image data...")
    start_time = time.time()

    try:
        # Log the inputs we're working with
        logging.info(f"Text summary length: {len(text_summary)} chars")
        logging.info(f"Number of image summaries: {len(image_summaries)}")

        # Prepare image summary text
        image_summary_text = "\n".join([f"Frame {i}: {summary}" for i, summary in enumerate(image_summaries)])
        logging.info(f"Combined {len(image_summaries)} image summaries")

        # Send to GPT for final analysis
        logging.info("Sending combined data to OpenAI for final analysis")
        prompt = f"""
        Analyze this video content:

        AUDIO TRANSCRIPT SUMMARY:
        {text_summary}

        KEY VISUAL ELEMENTS:
        {image_summary_text}

        Provide a comprehensive analysis with these sections:
        1. OVERVIEW: Summarize what's happening in this video
        2. KEY POINTS: List the most important details
        3. CONTEXT: Describe the setting and environment
        """

        response = openai.ChatCompletion.create(
            model="gpt-4",  # Using a more capable model for the final analysis
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000
        )

        result = response['choices'][0]['message']['content']
        elapsed_time = time.time() - start_time
        logging.info(f"Comprehensive analysis complete. Took {elapsed_time:.2f} seconds.")
        return result
    except Exception as e:
        logging.error(f"Error in comprehensive analysis: {str(e)}")
        raise
# ==============================
# MAIN FUNCTION
# ==============================
def main(video_path):
    try:
        print(f"🚀 Starting video analysis process for: {video_path}")
        logging.info(f"=== STARTING VIDEO ANALYSIS FOR: {video_path} ===")

        # Step 0: Validate input
        if not os.path.exists(video_path):
            logging.error(f"Video file not found: {video_path}")
            raise FileNotFoundError(f"Video file not found: {video_path}")

        print(f"📊 Video size: {os.path.getsize(video_path) / (1024*1024):.1f} MB")
        logging.info(f"Video file validated. Size: {os.path.getsize(video_path) / (1024*1024):.1f} MB")

        # Step 1: Extract key frames
        print(f"🔍 PHASE 1/5: Extracting key frames... (This may take a while)")
        extract_key_frames(video_path)

        # Verify frames were extracted
        frames = glob.glob("frames/*.jpg")
        if not frames:
            logging.error("Frame extraction failed - no frames were saved")
            raise RuntimeError("Frame extraction failed - no frames were produced")
        print(f"✅ Frame extraction complete. Extracted {len(frames)} key frames.")

        # Step 2: Extract audio
        print(f"🔊 PHASE 2/5: Extracting audio...")
        extract_audio(video_path)

        # Verify audio was extracted
        if not os.path.exists("audio.wav"):
            logging.error("Audio extraction failed - audio.wav not found")
            raise FileNotFoundError("Audio extraction failed - output file not found")
        print(f"✅ Audio extraction complete: {os.path.getsize('audio.wav') / (1024*1024):.1f} MB")

        # Step 3: Transcribe audio
        print(f"📝 PHASE 3/5: Transcribing audio... (This may take a while)")
        transcribe_audio("audio.wav")

        # Verify transcript was created
        transcript_file = "transcript.txt"
        if not os.path.exists(transcript_file):
            logging.error(f"Transcription failed - {transcript_file} not found")
            raise FileNotFoundError(f"Transcription failed - {transcript_file} not found")
        print(f"✅ Transcription complete: {os.path.getsize(transcript_file)} bytes")

        # Step 4: Analyze text
        print(f"📊 PHASE 4/5: Analyzing transcript...")
        summary = analyze_text(transcript_file)
        print(f"✅ Text analysis complete. Summary length: {len(summary)} chars")
        logging.info(f"Text analysis summary: {summary[:100]}...")

        # Step 5: Analyze images
        print(f"🖼️  PHASE 5/5: Analyzing images... (This will take a while)")
        frames = sorted(glob.glob("frames/*.jpg"))
        if not frames:
            logging.error("No frames found for image analysis")
            raise FileNotFoundError("No frames found for image analysis")
        print(f"   Found {len(frames)} frames to analyze")

        image_summaries = asyncio.run(process_all_frames())
        print(f"✅ Image analysis complete. Analyzed {len(image_summaries)} frames")

        # Step 6: Comprehensive analysis
        print(f"🧠 FINAL PHASE: Creating comprehensive analysis...")
        full_analysis_text = full_analysis(summary, image_summaries)
        print(f"✅ Comprehensive analysis complete. Length: {len(full_analysis_text)} chars")

        # Save results
        output_file = "summary.txt"
        with open(output_file, "w") as f:
            f.write(full_analysis_text)
        print(f"💾 Results saved to {output_file}")
        print("✅ Video analysis complete! Summary saved.")
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Cleaning up...")
        # Additional cleanup if needed
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
    except Exception as e:
        logging.error(f"Error in main process: {str(e)}")
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    import sys

    try:
        if len(sys.argv) < 2:
            print("Usage: python video_analyzer.py <video_file>")
            sys.exit(1)

        video_path = sys.argv[1]
        main(video_path)
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
