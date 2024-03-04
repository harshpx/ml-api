# All-in-one API for all my Machine Learning Models

This is a multipurpose API created using Python and FastAPI. <br/> This contains all my machine learning and deep learning models. Feel free to use!

This API is under constant development, more models and endpoints will be added soon!

#### <b>Tech Stack:</b> Python, FastAPI, Tensorflow, Scikit-Learn, OpenCV, NLTK etc.

## Models currently available

* ### Dog Breed Identification
    (A CNN based Image classification models, that classifies for 70 different dog breeds with an accuracy of ~95%)


## API Endpoints
* ### /dog-breed-identifier/upload <br/>
    (takes Image formData as input)<br/>
    Example JS code:
    ```
    const BASE_URL='https://mlapi.online';
    const file = event.target.files[0]; //image blob
    const formData = new FormData();
    formData.append('image', file);

    fetch(`${BASE_URL}/dog-breed-identifier/upload`,{
        method:"POST",
        body:formData
    })
    .then(res=>res.json())
    .then(res=>{
        //write further logic
    })
    .catch(error=>{
        console.log(error);
        //write further logic
    })
    ```

* ### /dog-breed-identifier/url <br/>
    (takes a vaild URL string as input)<br/>
    Example JS code:
    ```
    const BASE_URL='https://mlapi.online';
    fetch(`${BASE_URL}/dog-breed-identifier/url`,{
        method:"POST",
        headers: {
            'Content-Type': 'application/json'
        },			
        body:JSON.stringify({"url":"YOUR URL"})
    })
    .then(res=>res.json())
    .then(res=>{
        // further logic
    })
    .catch(error=>{
        // further logic
    })
    ```
