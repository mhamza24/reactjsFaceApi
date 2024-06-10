import { useRef, useEffect } from 'react';
import './App.css';
import * as faceapi from 'face-api.js';

function App() {
  const videoRef = useRef();
  const canvasRef = useRef();

  useEffect(() => {
    startVideo();
    loadModels();
  }, []);

  const startVideo = () => {
    navigator.mediaDevices.getUserMedia({ video: true })
      .then((currentStream) => {
        videoRef.current.srcObject = currentStream;
      })
      .catch((err) => {
        console.log(err);
      });
  };

  const loadModels = () => {
    Promise.all([
      faceapi.nets.tinyFaceDetector.loadFromUri('/models'),
      faceapi.nets.faceLandmark68Net.loadFromUri('/models'),
      faceapi.nets.faceRecognitionNet.loadFromUri('/models'),
      faceapi.nets.faceExpressionNet.loadFromUri('/models'),
      faceapi.nets.ssdMobilenetv1.loadFromUri('/models')
    ]).then(() => {
      faceMyDetect();
    });
  };

  const faceMyDetect = async () => {
    const labeledDescriptors = await loadLabeledImages();
    const faceMatcher = new faceapi.FaceMatcher(labeledDescriptors, 0.7);

    videoRef.current.addEventListener('play', () => {
      const canvas = canvasRef.current;
      faceapi.createCanvasFromMedia(videoRef.current);
      const displaySize = { width: videoRef.current.videoWidth, height: videoRef.current.videoHeight };
      faceapi.matchDimensions(canvas, displaySize);

      setInterval(async () => {
        const detections = await faceapi.detectAllFaces(videoRef.current, new faceapi.SsdMobilenetv1Options()).withFaceLandmarks().withFaceDescriptors();
        const resizedDetections = faceapi.resizeResults(detections, displaySize);
        canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);

        const results = resizedDetections.map(d => faceMatcher.findBestMatch(d.descriptor));

        results.forEach((result, i) => {
          console.log(`Detected: ${result.toString()}`);
          const box = resizedDetections[i].detection.box;
          const drawBox = new faceapi.draw.DrawBox(box, { label: result.toString() });
          drawBox.draw(canvas);
        });
      }, 100);
    });
  };

  const loadLabeledImages = () => {
    const labels = ['Prashant Kumar', 'Captain America', 'Tony Stark', 'Muhammad Hamza Khalid', 'Khuzaima Ansari'];
    return Promise.all(
      labels.map(async (label) => {
        const descriptions = [];
        for (let i = 1; i <= 2; i++) {
          const img = await faceapi.fetchImage(`/src/labeled_images/${label}/${i}.jpg`);
          const detections = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor();
          descriptions.push(detections.descriptor);
        }
        return new faceapi.LabeledFaceDescriptors(label, descriptions);
      })
    );
  };

  return (
    <div className="myapp">
      <h1>Face Detection</h1>
      <div className="appvideo">
        <video ref={videoRef} autoPlay muted width="940" height="650"></video>
        <canvas ref={canvasRef} className="appcanvas" />
      </div>
    </div>
  );
}

export default App;
