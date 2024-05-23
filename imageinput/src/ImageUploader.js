import React, { useState } from 'react';
import "./style.css";

const ImageUploader = () => {
  const [image, setImage] = useState(null);
  const [age, setAge] = useState(null);
  const [gender, setGender] = useState(null); // Add state for gender

  const handleImageChange = (e) => {
    setImage(e.target.files[0]);
  };

  const handleImageClick = () => {
    document.getElementById('fileInput').click();
  };

  const handleImageUpload = async () => {
    if (!image) {
      console.error('No image selected');
      return;
    }

    const formData = new FormData();
    formData.append('image', image);

    try {
      const response = await fetch('http://localhost:5000/upload', {
        method: 'POST',
        body: formData
      });
      const data = await response.json();
      console.log(data);
      setAge(data.age);
      setGender(data.gender); // Set the gender received from the server
    } catch (error) {
      console.error('Error:', error);
    }
  };

  return (
    <div className="container">
      <div className="input-container" onClick={handleImageClick}>
        <input id="fileInput" type="file" onChange={handleImageChange} />
        {!image && <div className="file-input-overlay">Choose File</div>}
        {image && (
          <div className="image-preview">
            <img src={URL.createObjectURL(image)} alt="Preview" />
          </div>
        )}
      </div>
      <button className="upload-btn" onClick={handleImageUpload}>Upload Image</button>
      {age && gender && ( // Check for both age and gender before displaying
        <div className="age-display">
          Predicted Age: {age}<br />  {/* Add a line break for better formatting */}
          Predicted Gender: {gender}
        </div>
      )}
    </div>
  );
};

export default ImageUploader;
