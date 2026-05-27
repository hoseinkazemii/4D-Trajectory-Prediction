using System.Collections;
using UnityEngine;

public enum SignalPersonGeneralState
{
    SwingAlignmentLoading,
    TrolleyAlignmentLoading,
    LoweringCableLoading,
    Loading,
    HoistingCableUnloading,
    SwingAlignmentUnloading,
    TrolleyAlignmentUnloading,
    LoweringCableUnloading,
    Unloading,
    Finish,
}

public enum SignalPersonState
{
    SwingAlignmentLoading50ft,
    SwingAlignmentLoading30ft,
    SwingAlignmentLoading20ft,
    SwingAlignmentLoading10ft,
    SwingAlignmentLoading5ft,
    SwingAlignmentLoading2ft,

    TrolleyAlignmentLoading50ft,
    TrolleyAlignmentLoading30ft,
    TrolleyAlignmentLoading20ft,
    TrolleyAlignmentLoading10ft,
    TrolleyAlignmentLoading5ft,
    TrolleyAlignmentLoading2ft,

    LoweringCableLoading50ft,
    LoweringCableLoading30ft,
    LoweringCableLoading20ft,
    LoweringCableLoading10ft,
    LoweringCableLoading5ft,
    LoweringCableLoading2ft,
  
    Loading,

    HoistingCableUnloading50ft,
    HoistingCableUnloading30ft,
    HoistingCableUnloading20ft,
    HoistingCableUnloading10ft,
    HoistingCableUnloading5ft,
    HoistingCableUnloading2ft,

    SwingAlignmentUnloading60ft,
    SwingAlignmentUnloading50ft,
    SwingAlignmentUnloading30ft,
    SwingAlignmentUnloading20ft,
    SwingAlignmentUnloading10ft,
    SwingAlignmentUnloading5ft,
    SwingAlignmentUnloading2ft,

    TrolleyAlignmentUnloading50ft,
    TrolleyAlignmentUnloading30ft,
    TrolleyAlignmentUnloading20ft,
    TrolleyAlignmentUnloading10ft,
    TrolleyAlignmentUnloading5ft,
    TrolleyAlignmentUnloading2ft,

    LoweringCableUnloading50ft,
    LoweringCableUnloading30ft,
    LoweringCableUnloading20ft,
    LoweringCableUnloading10ft,
    LoweringCableUnloading5ft,
    LoweringCableUnloading2ft,

    Unloading,

    Finish,
}

public class SignalPerson : MonoBehaviour
{
    // Define a delegate for callback function
    private delegate void LoadingCallback();

    private Transform grabbableObject;
    private Transform unloadingArea;
    private Transform obstacle;
    private LayerMask grabbableLayer;
    private LayerMask unloadingAreaLayer;
    private LayerMask obstacleLayer;
    private LayerMask firstFloorUnloadingBuildingLayer;
    [SerializeField] private AudioSource audioSource;

    private bool grabbableObjectInitialized = false;
    private bool gameStarted = false;

    private Transform craneHook;
    private Vector2 craneHookPos;
    private Vector2 grabbableObjectPos;
    private Vector2 unloadingAreaPos;
    private Vector3 cameraPosition;
    private Vector3 cameraPosOnXZ;
    private float signedAngularDistanceBwObjHook;
    private float signedAngularDistanceBwObjUnloading;
    private Ray ray;
    private RaycastHit hit;
    private float commandRepeatingTime = 15f;
    private float craneHookHeight;
    private float angularDistanceLimit = 5f;
    private float grabReleaseRange = 20f;
    private float obstacleMaximumHeightBoundary = 30f;
    private float obstacleBoundary;
    private bool stopAfterCableDownForUnloadingSaid = false;
    private bool stopAlreadySaid = false;

    private float distanceToCraneHook;
    private float distanceToUnloadingArea;
    private float distanceToGrabbableObject;
    private float horizontalDistanceBwHookObj;
    private float horizontalDistanceBwHookUnl;
    private float distanceAboveGrabbableObject;
    private float distanceAboveObstacle;
    private float distanceAboveUnloadingArea;

    private SignalPersonState currentState;
    private SignalPersonState previousState = SignalPersonState.Finish; // Initialize previousState to null
    private float lastCommandTime;

    // Define the thresholds for different angular distance commands
    float[] angularDistanceThresholdsLoading = { 50f, 30f, 20f, 10f, 5f, 2f };
    // Angular distance thresholds for unloading the object 
    float[] angularDistanceThresholdsUnloading = { 60f, 50f, 30f, 20f, 10f, 5f, 2f };
    // Define the thresholds for different trolley movement commands
    float[] trolleyMovementThresholds = { 50f, 30f, 20f, 10f, 5f, 2f };
    // Define the thresholds for different cable distance commands
    float[] cableDistanceThresholds = { 50f, 30f, 20f, 10f, 5f, 2f };

    void Start()
    {
        craneHook = transform;
        
        unloadingAreaLayer = 1 << LayerMask.NameToLayer("UnloadingArea");
        obstacleLayer = 1 << LayerMask.NameToLayer("Obstacle");
        firstFloorUnloadingBuildingLayer = 1 << LayerMask.NameToLayer("firstFloorUnloadingBuilding");

        // Find the unloadingArea in the scene based on its layer
        Collider[] unloadingColliders = Physics.OverlapSphere(Vector3.zero, Mathf.Infinity, unloadingAreaLayer);
        unloadingArea = unloadingColliders[0].transform;

        // Find the Obstacle in the scene based on its layer
        Collider[] obstacleColliders = Physics.OverlapSphere(Vector3.zero, Mathf.Infinity, obstacleLayer);
        obstacle = obstacleColliders[0].transform;
        // Raise the cable until the crane hook is obstacleMaximumHeightBoundary meters above the obstacle along the Y-axis
        Bounds obstacleBounds = new Bounds(obstacle.position, obstacle.localScale);
        obstacleBoundary = obstacleBounds.max.y + obstacleMaximumHeightBoundary;
    
        unloadingAreaPos = new Vector2(unloadingArea.position.x, unloadingArea.position.z);
    }

    // The following executes the actions for different Signal Person states in each frame
    void Update()
    {
        if (grabbableObjectInitialized) {
            InitializeGrabbableObject();
            grabbableObjectInitialized = false;
        }

        if (gameStarted){
            grabbableObjectPos = new Vector2(grabbableObject.position.x, grabbableObject.position.z);
            craneHookPos = new Vector2(craneHook.position.x, craneHook.position.z);

            // Measure angular distance between crane's hook and grabbable object's initial position
            signedAngularDistanceBwObjHook = Vector2.SignedAngle(craneHookPos, grabbableObjectPos)*2; // *2 is for adjusting- making angularDistance sound more real
            // Measure angular distance between crane's hook and unloading area's position
            signedAngularDistanceBwObjUnloading = Vector2.SignedAngle(craneHookPos, unloadingAreaPos);

            // Project Camera's position into X-Z plane
            cameraPosition = Camera.main.transform.position;
            cameraPosOnXZ = new Vector3(cameraPosition.x, cameraPosition.z);
            // Calculate distances in the X-Z plane
            distanceToCraneHook = Vector3.Distance(craneHookPos, cameraPosOnXZ);
            distanceToUnloadingArea = Vector3.Distance(unloadingAreaPos, cameraPosOnXZ);
            distanceToGrabbableObject = Vector3.Distance(grabbableObjectPos, cameraPosOnXZ);

            horizontalDistanceBwHookObj = distanceToGrabbableObject - distanceToCraneHook;
            horizontalDistanceBwHookUnl = distanceToUnloadingArea - distanceToCraneHook;

            // Draw a ray to see whether we are above the grabbable object
            ray = new Ray(transform.position, -transform.right);
            // Measuring crane hook height to compare it with obstacle's height
            craneHookHeight = craneHook.position.y;

            distanceAboveGrabbableObject = Mathf.Abs(craneHookHeight - grabbableObject.position.y);
            // Report the crane hook's distance above the obstacle
            distanceAboveObstacle = Mathf.Abs(craneHookHeight - obstacleBoundary);
            // Calculate distance above unloading area for "Cable Down" during unloading
            distanceAboveUnloadingArea = Mathf.Abs(craneHook.position.y - unloadingArea.position.y);

            // Call CurrentStateDesignator() function to get Signal Person's currentState
            currentState = CurrentStateDesignator();

            // Check if the general state has changed - Checking "Stop" command
            IsGeneralStateChanged(previousState, currentState);

            // Check if the current state is different from the previous one
            if (currentState != previousState)
            {
                // Send the command immediately when the state changes
                SendCommand();
            }
            // The following is for if currentState == previousState 
            else if (ShouldSendCommand())
            {
                // Send the command if it's time and the state hasn't changed
                SendCommand();
            }
        }
    }

    public void InitializeGrabbableObject()
    {
        grabbableLayer = 1 << LayerMask.NameToLayer("GrabbableObject");
        Collider[] grabbableColliders = Physics.OverlapSphere(Vector3.zero, Mathf.Infinity, grabbableLayer);
        if (grabbableColliders.Length > 0)
        {
            grabbableObject = grabbableColliders[0].transform;
        }
        gameStarted = true;
    }

    // Helper function to check if the general state has changed
    // This function will handle "Stop" command when general state is changed
    private void IsGeneralStateChanged(SignalPersonState previousState, SignalPersonState newState)
    {
        // Check if the previous state and new state belong to different general states
        if (GetGeneralState(previousState) != GetGeneralState(newState)){
            stopAlreadySaid = false;
        }
        else{
            stopAlreadySaid = true;
        }
    }

    // Helper function to determine the general state
    SignalPersonGeneralState GetGeneralState(SignalPersonState state)
    {
        switch (state)
        {
            case SignalPersonState.SwingAlignmentLoading50ft:
            case SignalPersonState.SwingAlignmentLoading30ft:
            case SignalPersonState.SwingAlignmentLoading20ft:
            case SignalPersonState.SwingAlignmentLoading10ft:
            case SignalPersonState.SwingAlignmentLoading5ft:
            case SignalPersonState.SwingAlignmentLoading2ft:
                return SignalPersonGeneralState.SwingAlignmentLoading;

            case SignalPersonState.TrolleyAlignmentLoading50ft:
            case SignalPersonState.TrolleyAlignmentLoading30ft:
            case SignalPersonState.TrolleyAlignmentLoading20ft:
            case SignalPersonState.TrolleyAlignmentLoading10ft:
            case SignalPersonState.TrolleyAlignmentLoading5ft:
            case SignalPersonState.TrolleyAlignmentLoading2ft:
                return SignalPersonGeneralState.TrolleyAlignmentLoading;

            case SignalPersonState.LoweringCableLoading50ft:
            case SignalPersonState.LoweringCableLoading30ft:
            case SignalPersonState.LoweringCableLoading20ft:
                return SignalPersonGeneralState.LoweringCableLoading;

            case SignalPersonState.LoweringCableLoading10ft:
            case SignalPersonState.LoweringCableLoading5ft:
            case SignalPersonState.LoweringCableLoading2ft:
            case SignalPersonState.Loading:
                return SignalPersonGeneralState.Loading;

            case SignalPersonState.HoistingCableUnloading50ft:
            case SignalPersonState.HoistingCableUnloading30ft:
            case SignalPersonState.HoistingCableUnloading20ft:
            case SignalPersonState.HoistingCableUnloading10ft:
            case SignalPersonState.HoistingCableUnloading5ft:
            case SignalPersonState.HoistingCableUnloading2ft:
                return SignalPersonGeneralState.HoistingCableUnloading;
            
            case SignalPersonState.SwingAlignmentUnloading60ft:
            case SignalPersonState.SwingAlignmentUnloading50ft:
            case SignalPersonState.SwingAlignmentUnloading30ft:
            case SignalPersonState.SwingAlignmentUnloading20ft:
            case SignalPersonState.SwingAlignmentUnloading10ft:
            case SignalPersonState.SwingAlignmentUnloading5ft:
            case SignalPersonState.SwingAlignmentUnloading2ft:
                return SignalPersonGeneralState.SwingAlignmentUnloading;

            case SignalPersonState.TrolleyAlignmentUnloading50ft:
            case SignalPersonState.TrolleyAlignmentUnloading30ft:
            case SignalPersonState.TrolleyAlignmentUnloading20ft:
            case SignalPersonState.TrolleyAlignmentUnloading10ft:
            case SignalPersonState.TrolleyAlignmentUnloading5ft:
            case SignalPersonState.TrolleyAlignmentUnloading2ft:
                return SignalPersonGeneralState.TrolleyAlignmentUnloading;

            case SignalPersonState.LoweringCableUnloading50ft:
            case SignalPersonState.LoweringCableUnloading30ft:
            case SignalPersonState.LoweringCableUnloading20ft:
                return SignalPersonGeneralState.LoweringCableUnloading;

            case SignalPersonState.LoweringCableUnloading10ft:
            case SignalPersonState.LoweringCableUnloading5ft:
            case SignalPersonState.LoweringCableUnloading2ft:
            case SignalPersonState.Unloading:
                return SignalPersonGeneralState.Unloading;
            
            case SignalPersonState.Finish:
                return SignalPersonGeneralState.Finish;


            default:
                // If the state is not part of any known general state, return the state itself
                return SignalPersonGeneralState.Finish;
        }
    }

    // Check if it's time to send a command
    bool ShouldSendCommand()
    {
        return Time.time - lastCommandTime >= commandRepeatingTime;
    }

    // Send the command based on the current state
    void SendCommand()
    {
        switch (currentState)
        {
            case SignalPersonState.SwingAlignmentLoading50ft:
            case SignalPersonState.SwingAlignmentLoading30ft:
            case SignalPersonState.SwingAlignmentLoading20ft:
            case SignalPersonState.SwingAlignmentLoading10ft:
            case SignalPersonState.SwingAlignmentLoading5ft:
            case SignalPersonState.SwingAlignmentLoading2ft:
                SwingLoading();
                break;

            case SignalPersonState.TrolleyAlignmentLoading50ft:
            case SignalPersonState.TrolleyAlignmentLoading30ft:
            case SignalPersonState.TrolleyAlignmentLoading20ft:
            case SignalPersonState.TrolleyAlignmentLoading10ft:
            case SignalPersonState.TrolleyAlignmentLoading5ft:
            case SignalPersonState.TrolleyAlignmentLoading2ft:
                TrolleyLoading();
                break;

            case SignalPersonState.LoweringCableLoading50ft:
            case SignalPersonState.LoweringCableLoading30ft:
            case SignalPersonState.LoweringCableLoading20ft:
            case SignalPersonState.LoweringCableLoading10ft:
                CableDownLoading();
                break;

            case SignalPersonState.LoweringCableLoading5ft:
            case SignalPersonState.LoweringCableLoading2ft:
            case SignalPersonState.Loading:
                LoadingGrabbableObject();
                break;

            case SignalPersonState.HoistingCableUnloading50ft:
            case SignalPersonState.HoistingCableUnloading30ft:
            case SignalPersonState.HoistingCableUnloading20ft:
            case SignalPersonState.HoistingCableUnloading10ft:
            case SignalPersonState.HoistingCableUnloading5ft:
            case SignalPersonState.HoistingCableUnloading2ft:
                CableUpUnloading();
                break;

            case SignalPersonState.SwingAlignmentUnloading60ft:
            case SignalPersonState.SwingAlignmentUnloading50ft:
            case SignalPersonState.SwingAlignmentUnloading30ft:
            case SignalPersonState.SwingAlignmentUnloading20ft:
            case SignalPersonState.SwingAlignmentUnloading10ft:
            case SignalPersonState.SwingAlignmentUnloading5ft:
            case SignalPersonState.SwingAlignmentUnloading2ft:
                SwingUnloading();
                break;

            case SignalPersonState.TrolleyAlignmentUnloading50ft:
            case SignalPersonState.TrolleyAlignmentUnloading30ft:
            case SignalPersonState.TrolleyAlignmentUnloading20ft:
            case SignalPersonState.TrolleyAlignmentUnloading10ft:
            case SignalPersonState.TrolleyAlignmentUnloading5ft:
            case SignalPersonState.TrolleyAlignmentUnloading2ft:
                TrolleyUnloading();
                break;

            case SignalPersonState.LoweringCableUnloading50ft:
            case SignalPersonState.LoweringCableUnloading30ft:
            case SignalPersonState.LoweringCableUnloading20ft:
                CableDownUnloading();
                break;

            case SignalPersonState.LoweringCableUnloading10ft:
            case SignalPersonState.LoweringCableUnloading5ft:
            case SignalPersonState.LoweringCableUnloading2ft:
            case SignalPersonState.Unloading:
                UnloadingGrabbableObject();
                break;

            case SignalPersonState.Finish:
                SilentSignalPerson();
                break;
        }

        // Update the time of the last command
        lastCommandTime = Time.time;
    }

    void SwingLoading()
    {
        // Check if any audio is currently playing
        if (audioSource.isPlaying)
        {
            // Stop any currently playing audio
            audioSource.Stop();
        }
        string angularDistanceCommand = GetAngularDistanceCommand(signedAngularDistanceBwObjHook, currentState);
        audioSource.PlayOneShot(GetAudioClip(angularDistanceCommand));
        previousState = currentState;
    }

    void SwingUnloading()
    {
        // Play stop when the Signal Person state is changed
        StartCoroutine(PlayStop(() =>
        {
            if (audioSource.isPlaying)
            {
                audioSource.Stop();
            }

            string angularDistanceCommand = GetAngularDistanceCommand(signedAngularDistanceBwObjUnloading, currentState);
            audioSource.PlayOneShot(GetAudioClip(angularDistanceCommand));
        }));
        previousState = currentState;
    }

    private string GetAngularDistanceCommand(float distance, SignalPersonState state)
    {
            string stateName = state.ToString(); // Get the name of the enum value
            // Extract the numeric part of the enum name (e.g., "50", "30", etc.)
            string distanceString = stateName.Substring(stateName.IndexOf("ft") - 2, 2);
            // Parse the numeric part to get the distance value

            if (distance < 0)
            {
                // Swing Right
                // Iterate through the thresholds and return the first one that is greater than the distance
                return "SwingRight" + distanceString + "ft";
            }
            else if (distance > 0)
            {
                // Swing Right
                // Iterate through the thresholds and return the first one that is greater than the distance
                return "SwingLeft" + distanceString + "ft";
            }
        // If none of the thresholds are met, return nothing
        return null;
    }

    void TrolleyLoading()
    {
        // Play stop when the Signal Person state is changed
        StartCoroutine(PlayStop(() =>
        {
            if (audioSource.isPlaying)
            {
                audioSource.Stop();
            }

            // Get the trolley movement command based on the distance
            string trolleyMovementCommand = GetTrolleyMovementCommand(distanceToCraneHook, distanceToGrabbableObject, currentState);
            audioSource.PlayOneShot(GetAudioClip(trolleyMovementCommand));
        }));
        previousState = currentState;
    }

    void TrolleyUnloading()
    {
        // Play stop when the Signal Person state is changed
        StartCoroutine(PlayStop(() =>
        {
            if (audioSource.isPlaying)
            {
                audioSource.Stop();
            }

            // Get the trolley movement command based on the distance
            string trolleyMovementCommand = GetTrolleyMovementCommand(distanceToCraneHook, distanceToUnloadingArea, currentState);
            audioSource.PlayOneShot(GetAudioClip(trolleyMovementCommand));
        }));
        previousState = currentState;
    }

    private string GetTrolleyMovementCommand(float distanceToCraneHook, float distanceToObjectOrUnloading, SignalPersonState state)
    {
        string stateName = state.ToString(); // Get the name of the enum value
        string distanceString = stateName.Substring(stateName.IndexOf("ft") - 2, 2);
        // Parse the numeric part to get the distance value

        if (distanceToCraneHook <= distanceToObjectOrUnloading)
        {
            // Trolley Out
            return "TrolleyOut" + distanceString + "ft";
        }
        // Grabbable object is behind the crane's hook, move trolley in
        else if (distanceToCraneHook > distanceToObjectOrUnloading)
        {
            // Trolley In
            return "TrolleyIn" + distanceString + "ft";
        }
        return null;
    }

    void CableDownLoading()
    {
        StartCoroutine(PlayStop(() =>
        {
            if (audioSource.isPlaying)
            {
                audioSource.Stop();
            }

            // Lower the cable until it is within 5 meters of the ground
            string cableDistanceCommand = GetCableDownDistanceCommand(distanceAboveGrabbableObject, currentState);
            audioSource.PlayOneShot(GetAudioClip(cableDistanceCommand));
        }));
        previousState = currentState;
    }

    void CableUpUnloading()
    {
        StartCoroutine(PlayStop(() =>
        {
            if (audioSource.isPlaying)
            {
                audioSource.Stop();
            }

            string cableDistanceCommand = GetCableUpDistanceCommand(distanceAboveObstacle, currentState);
            audioSource.PlayOneShot(GetAudioClip(cableDistanceCommand));
        }));
        previousState = currentState;
    }

    void CableDownUnloading()
    {
        StartCoroutine(PlayStop(() =>
        {
            if (audioSource.isPlaying)
            {
                audioSource.Stop();
            }

            // Lower the cable until it is within 5 meters of the unloadingArea
            // Report the crane hook's distance above the unloadingArea
            string cableDistanceCommand = GetCableDownDistanceCommand(distanceAboveUnloadingArea, currentState);
            audioSource.PlayOneShot(GetAudioClip(cableDistanceCommand));
        }));
        previousState = currentState;
    }

    private string GetCableDownDistanceCommand(float distance, SignalPersonState state)
    {
        string stateName = state.ToString(); // Get the name of the enum value
        string distanceString = stateName.Substring(stateName.IndexOf("ft") - 2, 2);
        // Parse the numeric part to get the distance value
        return "CableDown" + distanceString + "ft";
    }

    private string GetCableUpDistanceCommand(float distance, SignalPersonState state)
    {
        string stateName = state.ToString(); // Get the name of the enum value
        string distanceString = stateName.Substring(stateName.IndexOf("ft") - 2, 2);
        // Parse the numeric part to get the distance value
        return "CableUp" + distanceString + "ft";
    }

    // You can also use the following function for saying Stop while the Unloading is being started
    void LoadingGrabbableObject()
    {
        StartCoroutine(PlayStop(() =>
        {
            if (audioSource.isPlaying)
            {
                audioSource.Stop();
            }
            audioSource.PlayOneShot(GetAudioClip("AllClear"));
        }));
        previousState = currentState;
    }

    void UnloadingGrabbableObject()
    {
        StartCoroutine(PlayStop(() =>
        {
            if (audioSource.isPlaying)
            {
                audioSource.Stop();
            }
            audioSource.PlayOneShot(GetAudioClip("LetOfTheLoad"));
        }));
        previousState = currentState;
    }

    void SilentSignalPerson()
    {
        // This function intentionally left empty for Signal Person to do nothing
    }

    IEnumerator PlayStop(LoadingCallback callback)
    {
        if (!stopAlreadySaid)
        {
            // Check if any audio is currently playing
            if (audioSource.isPlaying)
            {
                // Stop any currently playing audio
                audioSource.Stop();
            }

            AudioClip audioClip = GetAudioClip("Stop");
            audioSource.PlayOneShot(audioClip);
            yield return new WaitForSeconds(audioClip.length + 1f); // Wait for the audio clip to finish playing (+1 second)
        }

        // At the end of PlayStop coroutine, call the callback function
        callback.Invoke();
    }

    private AudioClip GetAudioClip(string command)
    {
        // Access and return audio clip based on the command from the "Audio" folder
        // Assumes audio files are named correctly (e.g., SwingRight.mp3)
        string path = "Audio/" + command;

        return Resources.Load<AudioClip>(path);
    }

    SignalPersonState CurrentStateDesignator()
    {
        if (Physics.Raycast(ray, out hit, Mathf.Infinity, firstFloorUnloadingBuildingLayer) &&
                !grabbableObject.transform.IsChildOf(craneHook) &&
                !stopAfterCableDownForUnloadingSaid)
        {
            currentState = SignalPersonState.Finish;
        }

        // Check if alignment is needed
        else if (Mathf.Abs(signedAngularDistanceBwObjHook) > angularDistanceLimit && 
            !Physics.Raycast(ray, out hit, Mathf.Infinity, grabbableLayer) &&
            !grabbableObject.transform.IsChildOf(craneHook))
        {
            float signedAngularDistanceBwObjHookAbs = Mathf.Abs(signedAngularDistanceBwObjHook);

            if (signedAngularDistanceBwObjHookAbs > angularDistanceThresholdsLoading[0]){
                currentState = SignalPersonState.SwingAlignmentLoading50ft;
            }
            else if (signedAngularDistanceBwObjHookAbs > angularDistanceThresholdsLoading[1]){
                currentState = SignalPersonState.SwingAlignmentLoading30ft;
            }
            else if (signedAngularDistanceBwObjHookAbs > angularDistanceThresholdsLoading[2]){
                currentState = SignalPersonState.SwingAlignmentLoading20ft;
            }
            else if (signedAngularDistanceBwObjHookAbs > angularDistanceThresholdsLoading[3]){
                currentState = SignalPersonState.SwingAlignmentLoading10ft;
            }
            else if (signedAngularDistanceBwObjHookAbs > angularDistanceThresholdsLoading[4]){
                currentState = SignalPersonState.SwingAlignmentLoading5ft;
            }
            else if (signedAngularDistanceBwObjHookAbs > angularDistanceThresholdsLoading[5]){
                currentState = SignalPersonState.SwingAlignmentLoading2ft;
            }
            else{
                currentState = SignalPersonState.SwingAlignmentLoading2ft;
            }            
        } 

        else if (Mathf.Abs(signedAngularDistanceBwObjHook) < angularDistanceLimit && 
                !Physics.Raycast(ray, out hit, Mathf.Infinity, grabbableLayer) &&
                !grabbableObject.transform.IsChildOf(craneHook))
        {
            // Comparing distance between hook and (grabbableObject/unloadingArea) to change the current state 
            // immediately after command (50ft, 30ft, etc.) change
            horizontalDistanceBwHookObj = Mathf.Abs(horizontalDistanceBwHookObj);

            if (horizontalDistanceBwHookObj > trolleyMovementThresholds[0]){
                currentState = SignalPersonState.TrolleyAlignmentLoading50ft;
            }
            else if (horizontalDistanceBwHookObj > trolleyMovementThresholds[1]){
                currentState = SignalPersonState.TrolleyAlignmentLoading30ft;
            }
            else if (horizontalDistanceBwHookObj > trolleyMovementThresholds[2]){
                currentState = SignalPersonState.TrolleyAlignmentLoading20ft;
            }
            else if (horizontalDistanceBwHookObj > trolleyMovementThresholds[3]){
                currentState = SignalPersonState.TrolleyAlignmentLoading10ft;
            }
            else if (horizontalDistanceBwHookObj > trolleyMovementThresholds[4]){
                currentState = SignalPersonState.TrolleyAlignmentLoading5ft;
            }
            // You can erase the following "else if"; both are "else"
            else if (horizontalDistanceBwHookObj > trolleyMovementThresholds[5]){
                currentState = SignalPersonState.TrolleyAlignmentLoading2ft;
            }
            else{
                currentState = SignalPersonState.TrolleyAlignmentLoading2ft;
            }
        }

        else if (Mathf.Abs(signedAngularDistanceBwObjHook) < angularDistanceLimit && 
                Physics.Raycast(ray, out hit, Mathf.Infinity, grabbableLayer) && 
                !grabbableObject.transform.IsChildOf(craneHook) &&
                !Physics.Raycast(ray, out hit, Mathf.Infinity, unloadingAreaLayer))
        {
            if (distanceAboveGrabbableObject > cableDistanceThresholds[0]){
                currentState = SignalPersonState.LoweringCableLoading50ft;
            }
            else if (distanceAboveGrabbableObject > cableDistanceThresholds[1]){
                currentState = SignalPersonState.LoweringCableLoading30ft;
            }
            else if (distanceAboveGrabbableObject > cableDistanceThresholds[2]){
                currentState = SignalPersonState.LoweringCableLoading20ft;
            }
            else if (distanceAboveGrabbableObject > cableDistanceThresholds[3]){
                currentState = SignalPersonState.Loading;
            }
            else if (distanceAboveGrabbableObject > cableDistanceThresholds[4]){
                currentState = SignalPersonState.Loading;
            }
            // You can erase the following "else if"; both are "else"
            else if (distanceAboveGrabbableObject > cableDistanceThresholds[5]){
                currentState = SignalPersonState.Loading;
            }
            else{
                currentState = SignalPersonState.Loading;
            }   
        }

        // else if (Mathf.Abs(signedAngularDistanceBwObjHook) < angularDistanceLimit && 
        //         Physics.Raycast(ray, out hit, grabReleaseRange, grabbableLayer) && 
        //         !grabbableObject.transform.IsChildOf(craneHook) &&
        //         !Physics.Raycast(ray, out hit, Mathf.Infinity, unloadingAreaLayer))
        // {
        //     currentState = SignalPersonState.Loading;
        // }

        // Check if the grabbable object is attached to the crane hook and it is "BELOW" the obstacle
        else if (grabbableObject.transform.IsChildOf(craneHook) && 
                craneHookHeight < obstacleBoundary &&
                !Physics.Raycast(ray, out hit, Mathf.Infinity, firstFloorUnloadingBuildingLayer))
        {
            if (distanceAboveObstacle > cableDistanceThresholds[0]){
                currentState = SignalPersonState.HoistingCableUnloading50ft;
            }
            else if (distanceAboveObstacle > cableDistanceThresholds[1]){
                currentState = SignalPersonState.HoistingCableUnloading30ft;
            }
            else if (distanceAboveObstacle > cableDistanceThresholds[2]){
                currentState = SignalPersonState.HoistingCableUnloading20ft;
            }
            else if (distanceAboveObstacle > cableDistanceThresholds[3]){
                currentState = SignalPersonState.HoistingCableUnloading10ft;
            }
            else if (distanceAboveObstacle > cableDistanceThresholds[4]){
                currentState = SignalPersonState.HoistingCableUnloading5ft;
            }
            // You can erase the following "else if"; both are "else"
            else if (distanceAboveObstacle > cableDistanceThresholds[5]){
                currentState = SignalPersonState.HoistingCableUnloading2ft;
            }
            else{
                currentState = SignalPersonState.HoistingCableUnloading2ft;
            }   
        }

        // Check if the grabbable object is attached to the crane hook and it is "ABOVE" the obstacle
        else if (Mathf.Abs(signedAngularDistanceBwObjUnloading) > angularDistanceLimit &&
                grabbableObject.transform.IsChildOf(craneHook) && craneHookHeight > obstacleBoundary && 
                !Physics.Raycast(ray, out hit, Mathf.Infinity, unloadingAreaLayer))
        {
            float signedAngularDistanceBwObjUnloadingAbs = Mathf.Abs(signedAngularDistanceBwObjUnloading);

            if (signedAngularDistanceBwObjUnloadingAbs > angularDistanceThresholdsUnloading[0]){
                currentState = SignalPersonState.SwingAlignmentUnloading60ft;
            }
            else if (signedAngularDistanceBwObjUnloadingAbs > angularDistanceThresholdsUnloading[1]){
                currentState = SignalPersonState.SwingAlignmentUnloading50ft;
            }
            else if (signedAngularDistanceBwObjUnloadingAbs > angularDistanceThresholdsUnloading[2]){
                currentState = SignalPersonState.SwingAlignmentUnloading30ft;
            }
            else if (signedAngularDistanceBwObjUnloadingAbs > angularDistanceThresholdsUnloading[3]){
                currentState = SignalPersonState.SwingAlignmentUnloading20ft;
            }
            else if (signedAngularDistanceBwObjUnloadingAbs > angularDistanceThresholdsUnloading[4]){
                currentState = SignalPersonState.SwingAlignmentUnloading10ft;
            }
            else if (signedAngularDistanceBwObjUnloadingAbs > angularDistanceThresholdsUnloading[5]){
                currentState = SignalPersonState.SwingAlignmentUnloading5ft;
            }
            else if (signedAngularDistanceBwObjUnloadingAbs > angularDistanceThresholdsUnloading[6]){
                currentState = SignalPersonState.SwingAlignmentUnloading2ft;
            }
            else{
                currentState = SignalPersonState.SwingAlignmentUnloading2ft;
            }   
        }

        else if (Mathf.Abs(signedAngularDistanceBwObjUnloading) < angularDistanceLimit &&
                grabbableObject.transform.IsChildOf(craneHook) &&
                Mathf.Abs(signedAngularDistanceBwObjUnloading) < 5f &&
                !Physics.Raycast(ray, out hit, Mathf.Infinity, unloadingAreaLayer))
        {
            // Comparing distance between hook and (grabbableObject/unloadingArea) to change the current state 
            // immediately after command (50ft, 30ft, etc.) change
            horizontalDistanceBwHookUnl = Mathf.Abs(horizontalDistanceBwHookUnl);

            if (horizontalDistanceBwHookUnl > trolleyMovementThresholds[0]){
                currentState = SignalPersonState.TrolleyAlignmentUnloading50ft;
            }
            else if (horizontalDistanceBwHookUnl > trolleyMovementThresholds[1]){
                currentState = SignalPersonState.TrolleyAlignmentUnloading30ft;
            }
            else if (horizontalDistanceBwHookUnl > trolleyMovementThresholds[2]){
                currentState = SignalPersonState.TrolleyAlignmentUnloading20ft;
            }
            else if (horizontalDistanceBwHookUnl > trolleyMovementThresholds[3]){
                currentState = SignalPersonState.TrolleyAlignmentUnloading10ft;
            }
            else if (horizontalDistanceBwHookUnl > trolleyMovementThresholds[4]){
                currentState = SignalPersonState.TrolleyAlignmentUnloading5ft;
            }
            // You can erase the following "else if"; both are "else"
            else if (horizontalDistanceBwHookUnl > trolleyMovementThresholds[5]){
                currentState = SignalPersonState.TrolleyAlignmentUnloading2ft;
            }
            else{
                currentState = SignalPersonState.TrolleyAlignmentUnloading2ft;
            } 
        }

        else if (Mathf.Abs(signedAngularDistanceBwObjUnloading) < angularDistanceLimit &&
                Physics.Raycast(ray, out hit, Mathf.Infinity, unloadingAreaLayer) &&
                grabbableObject.transform.IsChildOf(craneHook))
        {
            if (distanceAboveUnloadingArea > cableDistanceThresholds[0]){
                currentState = SignalPersonState.LoweringCableLoading50ft;
            }
            else if (distanceAboveUnloadingArea > cableDistanceThresholds[1]){
                currentState = SignalPersonState.LoweringCableLoading30ft;
            }
            else if (distanceAboveUnloadingArea > cableDistanceThresholds[2]){
                currentState = SignalPersonState.LoweringCableLoading20ft;
            }
            else if (distanceAboveUnloadingArea > cableDistanceThresholds[3]){
                currentState = SignalPersonState.Unloading;
            }
            else if (distanceAboveUnloadingArea > cableDistanceThresholds[4]){
                currentState = SignalPersonState.Unloading;
            }
            // You can erase the following "else if"; both are "else"
            else if (distanceAboveUnloadingArea > cableDistanceThresholds[5]){
                currentState = SignalPersonState.Unloading;
            }
            else{
                currentState = SignalPersonState.Unloading;
            }              
        }

        else if (Mathf.Abs(signedAngularDistanceBwObjUnloading) < angularDistanceLimit && 
                Physics.Raycast(ray, out hit, grabReleaseRange, unloadingAreaLayer) && 
                grabbableObject.transform.IsChildOf(craneHook) &&
                !Physics.Raycast(ray, out hit, Mathf.Infinity, grabbableLayer))
        {
            currentState = SignalPersonState.Unloading;
        }
            
        return currentState;
        }
    }



// // First approach: Using Coroutine to simulate the signal person

// // using System;
// // using System.Collections;
// // using Unity.Loading;
// // using UnityEngine;

// // public class GroundPerson : MonoBehaviour
// // {
// //     private Transform grabbableObject;
// //     private Transform unloadingArea;
// //     private Transform obstacle;
// //     private LayerMask grabbableLayer;
// //     private LayerMask unloadingAreaLayer;
// //     private LayerMask obstacleLayer;
// //     private LayerMask firstFloorUnloadingBuildingLayer;
// //     [SerializeField] private AudioSource audioSource;

// //     private Transform craneHook;
// //     private Vector2 craneHookPos;
// //     private Vector2 grabbableObjectPos;
// //     private Vector2 unloadingAreaPos;
// //     private Vector3 cameraPosition;
// //     private Vector3 cameraPosOnXZ;
// //     private Ray ray;
// //     private RaycastHit hit;
// //     private float craneHookHeight;
// //     private float commandRepeatingTime = 2f;
// //     private float timeBetweenCommandAndDistance = 1f;
// //     private float angularDistanceLimit = 5.0f;
// //     private float obstacleMaximumHeightBoundary = 2f;
// //     private float obstacleBoundary;
// //     private bool stopAfterSwingRightOrLeftForLoadingAlreadySaid = false;
// //     private bool stopAfterTolleyInOrOutForLoadingAlreadySaid = false;
// //     private bool stopAfterCableUpForLoadingAlreadySaid = false;
// //     private bool stopAfterSwingRightOrLeftForUnloadingAlreadySaid = false;
// //     private bool stopAfterTolleyInOrOutForUnloadingAlreadySaid = false;
// //     private bool stopAfterCableDownForLoadingSaid = false;
// //     private bool stopAfterCableDownForUnloadingSaid = false;
// //     private bool operating = true;

// //     private void Start()
// //     {
// //         craneHook = transform;
// //         grabbableLayer = 1 << LayerMask.NameToLayer("GrabbableObject");
// //         unloadingAreaLayer = 1 << LayerMask.NameToLayer("UnloadingArea");
// //         obstacleLayer = 1 << LayerMask.NameToLayer("Obstacle");
// //         firstFloorUnloadingBuildingLayer = 1 << LayerMask.NameToLayer("firstFloorUnloadingBuilding");

// //         // Find the grabbableObject in the scene based on its layer
// //         Collider[] grabbableColliders = Physics.OverlapSphere(Vector3.zero, Mathf.Infinity, grabbableLayer);
// //         grabbableObject = grabbableColliders[0].transform;

// //         // Find the unloadingArea in the scene based on its layer
// //         Collider[] unloadingColliders = Physics.OverlapSphere(Vector3.zero, Mathf.Infinity, unloadingAreaLayer);
// //         unloadingArea = unloadingColliders[0].transform;

// //         // Find the Obstacle in the scene based on its layer
// //         Collider[] obstacleColliders = Physics.OverlapSphere(Vector3.zero, Mathf.Infinity, obstacleLayer);
// //         obstacle = obstacleColliders[0].transform;
    
// //         StartCoroutine(OperateCrane());
// //     }

// //     private void Update()
// //     {

// //     }

// //     private IEnumerator OperateCrane()
// //     {
// //         while (operating)
// //         {
// //             craneHookPos = new Vector2(craneHook.position.x, craneHook.position.z);
// //             grabbableObjectPos = new Vector2(grabbableObject.position.x, grabbableObject.position.z);
// //             unloadingAreaPos = new Vector2(unloadingArea.position.x, unloadingArea.position.z);

// //             // Measure angular distance between crane's hook and grabbable object's initial position
// //             float signedAngularDistanceBwObjHook = Vector2.SignedAngle(craneHookPos, grabbableObjectPos)*2;
// //             // Measure angular distance between crane's hook and unloading area's position
// //             float signedAngularDistanceBwObjUnloading = Vector2.SignedAngle(craneHookPos, unloadingAreaPos);

// //             // Project Camera's position into X-Z plane
// //             cameraPosition = Camera.main.transform.position;
// //             cameraPosOnXZ = new Vector3(cameraPosition.x, cameraPosition.z);

// //             // Draw a ray to see whether we are above the grabbable object
// //             ray = new Ray(transform.position, -transform.right);

// //             // Check if alignment is needed
// //             if (Mathf.Abs(signedAngularDistanceBwObjHook) > angularDistanceLimit && 
// //                 !Physics.Raycast(ray, out hit, Mathf.Infinity, grabbableLayer) &&
// //                 !grabbableObject.transform.IsChildOf(craneHook))
// //             {
// //                 // Check if the command is "Swing Right" or "Swing Left"
// //                 if (Mathf.Abs(signedAngularDistanceBwObjHook) < 60f)
// //                 {
// //                     if (signedAngularDistanceBwObjHook < 0)
// //                     {
// //                         // Command: Swing Right
// //                         audioSource.PlayOneShot(GetAudioClip("SwingRight"));
// //                     }
// //                     else if (signedAngularDistanceBwObjHook > 0)
// //                     {
// //                         // Command: Swing Left
// //                         audioSource.PlayOneShot(GetAudioClip("SwingLeft"));
// //                     }

// //                     yield return new WaitForSeconds(timeBetweenCommandAndDistance);

// //                     string angularDistanceCommand = GetAngularDistanceCommand(signedAngularDistanceBwObjHook);
// //                     audioSource.PlayOneShot(GetAudioClip(angularDistanceCommand));
// //                 }
// //                 else
// //                 {
// //                     // Adjust based on the sign of signedAngularDistance
// //                     if (signedAngularDistanceBwObjHook < 0)
// //                     {
// //                         // Command: Swing Right
// //                         audioSource.PlayOneShot(GetAudioClip("SwingRight"));
// //                     }
// //                     else if (signedAngularDistanceBwObjHook > 0)
// //                     {
// //                         // Command: Swing Left
// //                         audioSource.PlayOneShot(GetAudioClip("SwingLeft"));
// //                     }
// //                 }

// //                 stopAfterSwingRightOrLeftForLoadingAlreadySaid = false;
// //             } 

// //             else if (Mathf.Abs(signedAngularDistanceBwObjHook) < angularDistanceLimit && 
// //                     !Physics.Raycast(ray, out hit, Mathf.Infinity, grabbableLayer) &&
// //                     !grabbableObject.transform.IsChildOf(craneHook))
// //             {
// //                 if (!stopAfterSwingRightOrLeftForLoadingAlreadySaid)
// //                 {
// //                     audioSource.PlayOneShot(GetAudioClip("Stop"));
// //                     yield return new WaitForSeconds(commandRepeatingTime);
// //                     stopAfterSwingRightOrLeftForLoadingAlreadySaid = true;
// //                 }

// //                 // Calculate distances in the X-Z plane
// //                 float distanceToCraneHook = Vector3.Distance(craneHookPos, cameraPosOnXZ);
// //                 float distanceToGrabbableObject = Vector3.Distance(grabbableObjectPos, cameraPosOnXZ);

// //                 // Check if the grabbable object is ahead of or behind the crane's hook in the X-Z plane
// //                 if (distanceToCraneHook < distanceToGrabbableObject)
// //                 {
// //                     // Grabbable object is ahead of the crane's hook, move trolley out
// //                     audioSource.PlayOneShot(GetAudioClip("TrolleyOut"));
// //                 }
// //                 else
// //                 {
// //                     // Grabbable object is behind the crane's hook, move trolley in
// //                     audioSource.PlayOneShot(GetAudioClip("TrolleyIn"));
// //                 }

// //                 yield return new WaitForSeconds(timeBetweenCommandAndDistance);

// //                 // Get the trolley movement command based on the distance
// //                 string trolleyMovementCommand = GetTrolleyMovementCommand(Mathf.Abs(distanceToGrabbableObject - distanceToCraneHook));
// //                 audioSource.PlayOneShot(GetAudioClip(trolleyMovementCommand));

// //                 stopAfterTolleyInOrOutForLoadingAlreadySaid = false;
// //             }

// //             else if (Mathf.Abs(signedAngularDistanceBwObjHook) < angularDistanceLimit && 
// //                     Physics.Raycast(ray, out hit, Mathf.Infinity, grabbableLayer) && 
// //                     !grabbableObject.transform.IsChildOf(craneHook) &&
// //                     !Physics.Raycast(ray, out hit, Mathf.Infinity, unloadingAreaLayer))
// //             {
// //                 if (!stopAfterTolleyInOrOutForLoadingAlreadySaid)
// //                 {
// //                     audioSource.PlayOneShot(GetAudioClip("Stop"));
// //                     yield return new WaitForSeconds(commandRepeatingTime);
// //                     stopAfterTolleyInOrOutForLoadingAlreadySaid = true;
// //                 }
// //                 // Keep saying "Cable Down" until the distance is within the desired range
// //                 audioSource.PlayOneShot(GetAudioClip("CableDown"));
// //                 yield return new WaitForSeconds(timeBetweenCommandAndDistance);

// //                 // Lower the cable until it is within 5 meters of the ground
// //                 if (!grabbableObject.transform.IsChildOf(craneHook))
// //                 {
// //                     // Report the crane hook's distance above the grabbableObject
// //                     float distanceAboveGrabbableObject = Mathf.Abs(craneHook.position.y - grabbableObject.position.y);
// //                     string distanceCommand = GetCableDistanceCommand(distanceAboveGrabbableObject);

// //                     // If the distance is within the desired range, continue saying "Cable Down"
// //                     if (distanceAboveGrabbableObject <= 60f)
// //                     {
// //                         audioSource.PlayOneShot(GetAudioClip(distanceCommand));
// //                     }

// //                     yield return new WaitForSeconds(commandRepeatingTime);
// //                 }

// //                 stopAfterCableDownForLoadingSaid = false;
                
// //                 // Raise the cable until the crane hook is obstacleMaximumHeightBoundary meters above the obstacle along the Y-axis
// //                 craneHookHeight = craneHook.position.y;
// //                 Bounds obstacleBounds = new Bounds(obstacle.position, obstacle.localScale);
// //                 obstacleBoundary = obstacleBounds.max.y + obstacleMaximumHeightBoundary;
// //             }

// //             // Check if the grabbable object is attached to the crane hook and it is "BELOW" the obstacle
// //             else if (grabbableObject.transform.IsChildOf(craneHook) && 
// //                     craneHookHeight < obstacleBoundary)
// //             {
// //                 // Say "Stop" only once after the object is grabbed
// //                 if (!stopAfterCableDownForLoadingSaid)
// //                 {
// //                     audioSource.PlayOneShot(GetAudioClip("Stop"));
// //                     stopAfterCableDownForLoadingSaid = true;
// //                     yield return new WaitForSeconds(commandRepeatingTime);
// //                 }

// //                 // Report the crane hook's distance above the obstacle
// //                 float distanceAboveObstacle = Mathf.Abs(craneHookHeight - obstacleBoundary);
// //                 string distanceCommand = GetCableDistanceCommand(distanceAboveObstacle);
// //                 audioSource.PlayOneShot(GetAudioClip("CableUp"));
// //                 yield return new WaitForSeconds(commandRepeatingTime);

// //                 // If the distance is within the desired range, continue saying "Cable Up"
// //                 if (distanceAboveObstacle <= 60f)
// //                 {
// //                     audioSource.PlayOneShot(GetAudioClip(distanceCommand));
// //                 }

// //                 // Update the crane hook height in each iteration
// //                 craneHookHeight = craneHook.position.y;
// //                 stopAfterCableUpForLoadingAlreadySaid = false;
// //             }

// //             // Check if the grabbable object is attached to the crane hook and it is "ABOVE" the obstacle
// //             else if (Mathf.Abs(signedAngularDistanceBwObjUnloading) > angularDistanceLimit &&
// //                     grabbableObject.transform.IsChildOf(craneHook) && craneHookHeight > obstacleBoundary && 
// //                     !Physics.Raycast(ray, out hit, Mathf.Infinity, unloadingAreaLayer))
// //             {
// //                 if (!stopAfterCableUpForLoadingAlreadySaid)
// //                 {
// //                     audioSource.PlayOneShot(GetAudioClip("Stop"));
// //                     yield return new WaitForSeconds(commandRepeatingTime);
// //                     stopAfterCableUpForLoadingAlreadySaid = true;
// //                 }
             
// //                 // Check if the command is "Swing Right" or "Swing Left"
// //                 if (Mathf.Abs(signedAngularDistanceBwObjUnloading) < 60f)
// //                 {
// //                     if (signedAngularDistanceBwObjUnloading < 0)
// //                     {
// //                         // Command: Swing Right
// //                         audioSource.PlayOneShot(GetAudioClip("SwingRight"));
// //                     }
// //                     else if (signedAngularDistanceBwObjUnloading > 0)
// //                     {
// //                         // Command: Swing Left
// //                         audioSource.PlayOneShot(GetAudioClip("SwingLeft"));
// //                     }

// //                     yield return new WaitForSeconds(timeBetweenCommandAndDistance);

// //                     string angularDistanceCommand = GetAngularDistanceCommand(signedAngularDistanceBwObjUnloading);
// //                     audioSource.PlayOneShot(GetAudioClip(angularDistanceCommand));
// //                 }
// //                 else
// //                 {
// //                     // Adjust based on the sign of signedAngularDistanceBwObjUnloading
// //                     if (signedAngularDistanceBwObjUnloading < 0)
// //                     {
// //                         // Command: Swing Right
// //                         audioSource.PlayOneShot(GetAudioClip("SwingRight"));
// //                     }
// //                     else if (signedAngularDistanceBwObjUnloading > 0)
// //                     {
// //                         // Command: Swing Left
// //                         audioSource.PlayOneShot(GetAudioClip("SwingLeft"));
// //                     }
// //                 }

// //                 stopAfterSwingRightOrLeftForUnloadingAlreadySaid = false;
// //             }

// //             else if (Mathf.Abs(signedAngularDistanceBwObjUnloading) < angularDistanceLimit &&
// //                     grabbableObject.transform.IsChildOf(craneHook) &&
// //                     Mathf.Abs(signedAngularDistanceBwObjUnloading) < 5f &&
// //                     !Physics.Raycast(ray, out hit, Mathf.Infinity, unloadingAreaLayer))
// //             {
// //                 if (!stopAfterSwingRightOrLeftForUnloadingAlreadySaid)
// //                 {
// //                     audioSource.PlayOneShot(GetAudioClip("Stop"));
// //                     yield return new WaitForSeconds(commandRepeatingTime);
// //                     stopAfterSwingRightOrLeftForUnloadingAlreadySaid = true;
// //                 }

// //                 // Calculate distances in the X-Z plane
// //                 float distanceToCraneHook = Vector3.Distance(craneHookPos, cameraPosOnXZ);
// //                 float distanceToUnloadingArea = Vector3.Distance(unloadingAreaPos, cameraPosOnXZ);

// //                 // Check if the unloading area is ahead of or behind the crane's hook in the X-Z plane
// //                 if (distanceToCraneHook < distanceToUnloadingArea)
// //                 {
// //                     // Grabbable object is ahead of the crane's hook, move trolley out
// //                     audioSource.PlayOneShot(GetAudioClip("TrolleyOut"));
// //                 }
// //                 else
// //                 {
// //                     // Grabbable object is behind the crane's hook, move trolley in
// //                     audioSource.PlayOneShot(GetAudioClip("TrolleyIn"));
// //                 }

// //                 yield return new WaitForSeconds(timeBetweenCommandAndDistance);

// //                 // Get the trolley movement command based on the distance
// //                 string trolleyMovementCommand = GetTrolleyMovementCommand(Mathf.Abs(distanceToUnloadingArea - distanceToCraneHook));
// //                 audioSource.PlayOneShot(GetAudioClip(trolleyMovementCommand));

// //                 stopAfterTolleyInOrOutForUnloadingAlreadySaid = false;      
// //             }

// //             else if (Mathf.Abs(signedAngularDistanceBwObjUnloading) < angularDistanceLimit &&
// //                     Physics.Raycast(ray, out hit, Mathf.Infinity, unloadingAreaLayer) &&
// //                     grabbableObject.transform.IsChildOf(craneHook))
// //             {
// //                 if (!stopAfterTolleyInOrOutForUnloadingAlreadySaid)
// //                 {
// //                     audioSource.PlayOneShot(GetAudioClip("Stop"));
// //                     yield return new WaitForSeconds(commandRepeatingTime);
// //                     stopAfterTolleyInOrOutForUnloadingAlreadySaid = true;
// //                 }

// //                 // Keep saying "Cable Down" until the distance is within the desired range
// //                 audioSource.PlayOneShot(GetAudioClip("CableDown"));
// //                 yield return new WaitForSeconds(timeBetweenCommandAndDistance);

// //                 // Lower the cable until it is within 5 meters of the unloadingArea
// //                 // Report the crane hook's distance above the unloadingArea
// //                 float distanceAboveUnloadingArea = Mathf.Abs(craneHook.position.y - unloadingArea.position.y);
// //                 string distanceCommand = GetCableDistanceCommand(distanceAboveUnloadingArea);

// //                 // If the distance is within the desired range, continue saying "Cable Down"
// //                 if (distanceAboveUnloadingArea <= 60f)
// //                 {
// //                     audioSource.PlayOneShot(GetAudioClip(distanceCommand));
// //                 }

// //                 stopAfterCableDownForUnloadingSaid = false;
// //             }

// //             else if (Mathf.Abs(signedAngularDistanceBwObjUnloading) < angularDistanceLimit &&
// //                     Physics.Raycast(ray, out hit, Mathf.Infinity, firstFloorUnloadingBuildingLayer) &&
// //                     !grabbableObject.transform.IsChildOf(craneHook) &&
// //                     !stopAfterCableDownForUnloadingSaid)
// //             {
// //                 audioSource.PlayOneShot(GetAudioClip("Stop"));
// //                 yield return new WaitForSeconds(commandRepeatingTime);
// //                 stopAfterCableDownForUnloadingSaid = true;

// //                 // Operation is finished
// //                 operating = false;
// //             }

// //             // Wait for "commandRepeatingTime" seconds before repeating the command
// //             yield return new WaitForSeconds(commandRepeatingTime);
// //         }
// //     }


// //     private AudioClip GetAudioClip(string command)
// //     {
// //         // Access and return audio clip based on the command from the "Audio" folder
// //         // Assumes audio files are named correctly (e.g., SwingRight.mp3)
// //         string path = "Audio/" + command;

// //         return Resources.Load<AudioClip>(path);
// //     }

// //     private string GetAngularDistanceCommand(float distance)
// //     {
// //         // Define the thresholds for different angular distance commands
// //         float[] angularDistanceThresholds = { 50f, 30f, 20f, 10f, 5f };

// //         // Iterate through the thresholds and return the first one that is greater than the distance
// //         foreach (float threshold in angularDistanceThresholds)
// //         {
// //             if (Mathf.Abs(distance) > threshold)
// //             {
// //                 return threshold + "ft";
// //             }
// //         }

// //         // If no threshold is met, default to "Stop"
// //         return "Stop";
// //     }

// //     private string GetTrolleyMovementCommand(float distance)
// //     {
// //         float[] trolleyMovementThresholds = { 50f, 30f, 20f, 10f, 5f };

// //         // Iterate through the thresholds and return the first one that is greater than the distance
// //         foreach (float threshold in trolleyMovementThresholds)
// //         {
// //             if (distance > threshold)
// //             {
// //                 return threshold + "ft";
// //             }
// //         }

// //         // If no threshold is met, default to "Stop"
// //         return "Stop";
// //     }

// //     private string GetCableDistanceCommand(float distance)
// //     {
// //         // Define the thresholds for different distance commands
// //         float[] distanceThresholds = { 50f, 30f, 20f, 10f, 5f };

// //         // Iterate through the thresholds and return the first one that is greater than the distance
// //         foreach (float threshold in distanceThresholds)
// //         {
// //             if (distance > threshold)
// //             {
// //                 return threshold + "ft";
// //             }
// //         }

// //         // If no threshold is met, default to "Stop"
// //         stopAfterCableDownForLoadingSaid = true;
// //         stopAfterCableUpForLoadingAlreadySaid = true;
// //         return "Stop";
// //     }
// // }


// //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// // // Second approach: Using State Machine to simulate the signal person

// // using System;
// // using System.Collections;
// // using Unity.Loading;
// // using Unity.VisualScripting;
// // using UnityEngine;

// // public enum SignalPersonState
// // {
// //     SwingAlignmentLoading,
// //     TrolleyAlignmentLoading,
// //     LoweringCableLoading,
// //     Loading,
// //     HoistingCableUnloading,
// //     SwingAlignmentUnloading,
// //     TrolleyAlignmentUnloading,
// //     LoweringCableUnloading,
// //     Unloading,
// //     Finish,
// // }

// // public class SignalPerson : MonoBehaviour
// // {
// //     // Define a delegate for callback function
// //     private delegate void LoadingCallback();

// //     private Transform grabbableObject;
// //     private Transform unloadingArea;
// //     private Transform obstacle;
// //     private LayerMask grabbableLayer;
// //     private LayerMask unloadingAreaLayer;
// //     private LayerMask obstacleLayer;
// //     private LayerMask firstFloorUnloadingBuildingLayer;
// //     [SerializeField] private AudioSource audioSource;

// //     private Transform craneHook;
// //     private Vector2 craneHookPos;
// //     private Vector2 grabbableObjectPos;
// //     private Vector2 unloadingAreaPos;
// //     private Vector3 cameraPosition;
// //     private Vector3 cameraPosOnXZ;
// //     private float signedAngularDistanceBwObjHook;
// //     private float signedAngularDistanceBwObjUnloading;
// //     private Ray ray;
// //     private RaycastHit hit;
// //     private float craneHookHeight;
// //     private float angularDistanceLimit = 5.0f;
// //     private float grabReleaseRange = 5f;
// //     private float obstacleMaximumHeightBoundary = 20f;
// //     private float obstacleBoundary;
// //     private float commandRepeatingTime =15f;
// //     private bool stopAfterCableDownForUnloadingSaid = false;
// //     private bool stopAlreadySaid = false;

// //     private float distanceToCraneHook;
// //     private float distanceToUnloadingArea;
// //     private float distanceToGrabbableObject;

// //     private SignalPersonState currentState;
// //     private SignalPersonState? previousState = null; // Initialize previousState to null
// //     private float lastCommandTime;

// //     void Start()
// //     {
// //         craneHook = transform;
// //         grabbableLayer = 1 << LayerMask.NameToLayer("GrabbableObject");
// //         unloadingAreaLayer = 1 << LayerMask.NameToLayer("UnloadingArea");
// //         obstacleLayer = 1 << LayerMask.NameToLayer("Obstacle");
// //         firstFloorUnloadingBuildingLayer = 1 << LayerMask.NameToLayer("firstFloorUnloadingBuilding");

// //         unloadingAreaPos = new Vector2(unloadingArea.position.x, unloadingArea.position.z);

// //         // Find the grabbableObject in the scene based on its layer
// //         Collider[] grabbableColliders = Physics.OverlapSphere(Vector3.zero, Mathf.Infinity, grabbableLayer);
// //         grabbableObject = grabbableColliders[0].transform;

// //         // Find the unloadingArea in the scene based on its layer
// //         Collider[] unloadingColliders = Physics.OverlapSphere(Vector3.zero, Mathf.Infinity, unloadingAreaLayer);
// //         unloadingArea = unloadingColliders[0].transform;

// //         // Find the Obstacle in the scene based on its layer
// //         Collider[] obstacleColliders = Physics.OverlapSphere(Vector3.zero, Mathf.Infinity, obstacleLayer);
// //         obstacle = obstacleColliders[0].transform;
// //         // Raise the cable until the crane hook is obstacleMaximumHeightBoundary meters above the obstacle along the Y-axis
// //         Bounds obstacleBounds = new Bounds(obstacle.position, obstacle.localScale);
// //         obstacleBoundary = obstacleBounds.max.y + obstacleMaximumHeightBoundary;
// //     }

// //     // The following executes the actions for different Signal Person states in each frame
// //     void Update()
// //     {
// //         craneHookPos = new Vector2(craneHook.position.x, craneHook.position.z);
// //         grabbableObjectPos = new Vector2(grabbableObject.position.x, grabbableObject.position.z);

// //         // Measure angular distance between crane's hook and grabbable object's initial position
// //         signedAngularDistanceBwObjHook = Vector2.SignedAngle(craneHookPos, grabbableObjectPos)*2;
// //         // Measure angular distance between crane's hook and unloading area's position
// //         signedAngularDistanceBwObjUnloading = Vector2.SignedAngle(craneHookPos, unloadingAreaPos);

// //         // Project Camera's position into X-Z plane
// //         cameraPosition = Camera.main.transform.position;
// //         cameraPosOnXZ = new Vector3(cameraPosition.x, cameraPosition.z);
// //         // Calculate distances in the X-Z plane
// //         distanceToCraneHook = Vector3.Distance(craneHookPos, cameraPosOnXZ);
// //         distanceToUnloadingArea = Vector3.Distance(unloadingAreaPos, cameraPosOnXZ);
// //         distanceToGrabbableObject = Vector3.Distance(grabbableObjectPos, cameraPosOnXZ);
// //         // Draw a ray to see whether we are above the grabbable object
// //         ray = new Ray(transform.position, -transform.right);
// //         // Measuring crane hook height to compare it with obstacle's height
// //         craneHookHeight = craneHook.position.y;

// //         // Call CurrentStateDesignator() function to get Signal Person's currentState
// //         currentState = CurrentStateDesignator();

// //         // Check if the current state is different from the previous one
// //         if (currentState != previousState)
// //         {
// //             // Send the command immediately when the state changes
// //             stopAlreadySaid = false;
// //             SendCommand();
// //         }
// //         // The following is for if currentState == previousState 
// //         else if (ShouldSendCommand())
// //         {
// //             // Send the command if it's time and the state hasn't changed
// //             stopAlreadySaid = true;
// //             SendCommand();
// //         }
// //     }

// //     // Check if it's time to send a command
// //     bool ShouldSendCommand()
// //     {
// //         return Time.time - lastCommandTime >= commandRepeatingTime;
// //     }

// //     // Send the command based on the current state
// //     void SendCommand()
// //     {
// //         switch (currentState)
// //         {
// //             case SignalPersonState.SwingAlignmentLoading:
// //                 SwingLoading();
// //                 break;

// //             case SignalPersonState.TrolleyAlignmentLoading:
// //                 TrolleyLoading();
// //                 break;

// //             case SignalPersonState.LoweringCableLoading:
// //                 CableDownLoading();
// //                 break;

// //             case SignalPersonState.Loading:
// //                 LoadingGrabbableObject();
// //                 break;

// //             case SignalPersonState.HoistingCableUnloading:
// //                 CableUpUnloading();
// //                 break;

// //             case SignalPersonState.SwingAlignmentUnloading:
// //                 SwingUnloading();
// //                 break;
            
// //             case SignalPersonState.TrolleyAlignmentUnloading:
// //                 TrolleyUnloading();
// //                 break;

// //             case SignalPersonState.LoweringCableUnloading:
// //                 CableDownUnloading();
// //                 break;
            
// //             case SignalPersonState.Unloading:
// //                 LoadingGrabbableObject();
// //                 break;

// //             case SignalPersonState.Finish:
// //                 // Write the function for finishing unloading here
// //                 break;
// //         }

// //         // Update the time of the last command
// //         lastCommandTime = Time.time;
// //     }

// //     void SwingLoading()
// //     {
// //         string angularDistanceCommand = GetAngularDistanceCommand(signedAngularDistanceBwObjHook);
// //         audioSource.PlayOneShot(GetAudioClip(angularDistanceCommand));
// //         previousState = currentState;
// //     }

// //     void TrolleyLoading()
// //     {
// //         // Play stop when the Signal Person state is changed
// //         StartCoroutine(PlayStop(() =>
// //         {
// //             // Get the trolley movement command based on the distance
// //             string trolleyMovementCommand = GetTrolleyMovementCommand(distanceToCraneHook, distanceToGrabbableObject);
// //             audioSource.PlayOneShot(GetAudioClip(trolleyMovementCommand));
// //         }));
// //         previousState = currentState;
// //     }

// //     void CableDownLoading()
// //     {
// //         StartCoroutine(PlayStop(() =>
// //         {
// //             // Lower the cable until it is within 5 meters of the ground
// //             float distanceAboveGrabbableObject = Mathf.Abs(craneHook.position.y - grabbableObject.position.y);
// //             string cableDistanceCommand = GetCableDownDistanceCommand(distanceAboveGrabbableObject);
// //             audioSource.PlayOneShot(GetAudioClip(cableDistanceCommand));
// //         }));
// //         previousState = currentState;
// //     }

// //     void CableUpUnloading()
// //     {
// //         StartCoroutine(PlayStop(() =>
// //         {
// //             // Report the crane hook's distance above the obstacle
// //             float distanceAboveObstacle = Mathf.Abs(craneHookHeight - obstacleBoundary);
// //             string cableDistanceCommand = GetCableUpDistanceCommand(distanceAboveObstacle);
// //             audioSource.PlayOneShot(GetAudioClip(cableDistanceCommand));
// //         }));
// //         previousState = currentState;
// //     }    

// //     // You can also use the following function for saying Stop while the Unloading is being started
// //     void LoadingGrabbableObject()
// //     {
// //         StartCoroutine(PlayStop(() => { }));
// //         previousState = currentState;
// //     }

// //     void SwingUnloading()
// //     {
// //         StartCoroutine(PlayStop(() =>
// //         {
// //             string angularDistanceCommand = GetAngularDistanceCommand(signedAngularDistanceBwObjUnloading);
// //             audioSource.PlayOneShot(GetAudioClip(angularDistanceCommand));
// //         }));
// //         previousState = currentState;
// //     }

// //     void TrolleyUnloading()
// //     {
// //         // Play stop when the Signal Person state is changed
// //         StartCoroutine(PlayStop(() =>
// //         {
// //             // Get the trolley movement command based on the distance
// //             string trolleyMovementCommand = GetTrolleyMovementCommand(distanceToCraneHook, distanceToUnloadingArea);
// //             audioSource.PlayOneShot(GetAudioClip(trolleyMovementCommand));
// //         }));
// //         previousState = currentState;
// //     }

// //     void CableDownUnloading()
// //     {
// //         StartCoroutine(PlayStop(() =>
// //         {
// //             // Lower the cable until it is within 5 meters of the unloadingArea
// //             // Report the crane hook's distance above the unloadingArea
// //             float distanceAboveUnloadingArea = Mathf.Abs(craneHook.position.y - unloadingArea.position.y);
// //             string cableDistanceCommand = GetCableDownDistanceCommand(distanceAboveUnloadingArea);
// //             audioSource.PlayOneShot(GetAudioClip(cableDistanceCommand));
// //         }));
// //         previousState = currentState;
// //     }

// //     IEnumerator PlayStop(LoadingCallback callback)
// //     {
// //         if (!stopAlreadySaid)
// //         {
// //             // Check if any audio is currently playing
// //             if (audioSource.isPlaying)
// //             {
// //                 // Stop any currently playing audio
// //                 audioSource.Stop();
// //             }

// //             AudioClip audioClip = GetAudioClip("Stop");
// //             audioSource.PlayOneShot(audioClip);
// //             yield return new WaitForSeconds(audioClip.length + 1f); // Wait for the audio clip to finish playing (+1 second)
// //         }

// //         // At the end of PlayStop coroutine, call the callback function- Playing Stop first; then, playing the command
// //         callback.Invoke();
// //     }

// //     private AudioClip GetAudioClip(string command)
// //     {
// //         // Access and return audio clip based on the command from the "Audio" folder
// //         // Assumes audio files are named correctly (e.g., SwingRight.mp3)
// //         string path = "Audio/" + command;

// //         return Resources.Load<AudioClip>(path);
// //     }

// //     private string GetAngularDistanceCommand(float distance)
// //     {
// //         // Define the thresholds for different angular distance commands
// //         float[] angularDistanceThresholds = { 50f, 30f, 20f, 10f, 5f, 2f };

// //             if (distance < 0)
// //             {
// //                 // Swing Right
// //                 // Iterate through the thresholds and return the first one that is greater than the distance
// //                 foreach (float threshold in angularDistanceThresholds)
// //                 {
// //                     if (MathF.Abs(distance) > threshold)
// //                     {
// //                         return "SwingRight" + threshold + "ft";
// //                     }

// //                     if (MathF.Abs(distance) < 2f)
// //                     {
// //                         return "SwingRight2ft";
// //                     }
// //                 }
// //             }
// //             else if (distance > 0)
// //             {
// //                 // Swing Left
// //                 // Iterate through the thresholds and return the first one that is greater than the distance
// //                 foreach (float threshold in angularDistanceThresholds)
// //                 {
// //                     if (distance > threshold)
// //                     {
// //                         return "SwingLeft" + threshold + "ft";
// //                     }

// //                     if (distance < 2f)
// //                     {
// //                         return "SwingLeft2ft";
// //                     }
// //                 }
// //             }
// //         // If none of the thresholds are met, return nothing
// //         return null;
// //     }

// //     private string GetTrolleyMovementCommand(float distanceToCraneHook, float distanceToObjectOrUnloading)
// //     {
// //         float[] trolleyMovementThresholds = { 50f, 30f, 20f, 10f, 5f, 2f };
        
// //         float distance = Mathf.Abs(distanceToObjectOrUnloading - distanceToCraneHook);
// //         // Check if the grabbable object is ahead of or behind the crane's hook in the X-Z plane
// //         // Grabbable object is ahead of the crane's hook, move trolley out
// //         if (distanceToCraneHook <= distanceToObjectOrUnloading)
// //         {
// //             // Trolley Out
// //             foreach (float threshold in trolleyMovementThresholds)
// //             {
// //                 if (distance > threshold)
// //                 {
// //                     return "TrolleyOut" + threshold + "ft";
// //                 }
// //             }

// //             if (distance < 2f)
// //             {
// //             // If no threshold is met, set the default to 2
// //             return "TrolleyOut2ft";
// //             }
// //         }
// //         // Grabbable object is behind the crane's hook, move trolley in
// //         else if (distanceToCraneHook > distanceToObjectOrUnloading)
// //         {
// //             // Trolley In
// //             foreach (float threshold in trolleyMovementThresholds)
// //             {
// //                 if (distance > threshold)
// //                 {
// //                     return "TrolleyIn" + threshold + "ft";
// //                 }
// //             }
// //             if (distance < 2f)
// //             {
// //             // If no threshold is met, set the default to 2
// //             return "TrolleyIn2ft";
// //             }
// //         }

// //         return null;
// //     }

// //     private string GetCableDownDistanceCommand(float distance)
// //     {
// //         // Define the thresholds for different distance commands
// //         float[] distanceThresholds = { 50f, 30f, 20f, 10f, 5f, 2f };

// //         // Iterate through the thresholds and return the first one that is greater than the distance
// //         foreach (float threshold in distanceThresholds)
// //         {
// //             if (distance > threshold)
// //             {
// //                 return "CableDown" + threshold + "ft";
// //             }
// //         }
    
// //         if (distance < 2f)
// //         {
// //             // When the distance is between 0 ft and 2 ft
// //             return "CableDown2ft";
// //         }

// //         // If none of the above conditions are met (one will be met!), don't send any commands
// //         return null;
// //     }

// //     private string GetCableUpDistanceCommand(float distance)
// //     {
// //         // Define the thresholds for different distance commands
// //         float[] distanceThresholds = { 50f, 30f, 20f, 10f, 5f, 2f };

// //         // Iterate through the thresholds and return the first one that is greater than the distance
// //         foreach (float threshold in distanceThresholds)
// //         {
// //             if (distance > threshold)
// //             {
// //                 return "CableUp" + threshold + "ft";
// //             }
// //         }

// //         if (distance < 2f)
// //         {
// //             // When the distance is between 0 ft and 2 ft
// //             return "CableUp2ft";
// //         }

// //         // If none of the above conditions are met (one will be met!), don't send any commands
// //         return null;
// //     }

// //     SignalPersonState CurrentStateDesignator()
// //     {
// //             // Check if alignment is needed
// //             if (Mathf.Abs(signedAngularDistanceBwObjHook) > angularDistanceLimit && 
// //                 !Physics.Raycast(ray, out hit, Mathf.Infinity, grabbableLayer) &&
// //                 !grabbableObject.transform.IsChildOf(craneHook))
// //             {
// //                 currentState = SignalPersonState.SwingAlignmentLoading;
// //             } 

// //             else if (Mathf.Abs(signedAngularDistanceBwObjHook) < angularDistanceLimit && 
// //                     !Physics.Raycast(ray, out hit, Mathf.Infinity, grabbableLayer) &&
// //                     !grabbableObject.transform.IsChildOf(craneHook))
// //             {
// //                 currentState = SignalPersonState.TrolleyAlignmentLoading;
// //             }

// //             else if (Mathf.Abs(signedAngularDistanceBwObjHook) < angularDistanceLimit && 
// //                     Physics.Raycast(ray, out hit, Mathf.Infinity, grabbableLayer) && 
// //                     !grabbableObject.transform.IsChildOf(craneHook) &&
// //                     !Physics.Raycast(ray, out hit, Mathf.Infinity, unloadingAreaLayer))
// //             {
// //                 currentState = SignalPersonState.LoweringCableLoading;
// //             }

// //             else if (Mathf.Abs(signedAngularDistanceBwObjHook) < angularDistanceLimit && 
// //                     Physics.Raycast(ray, out hit, grabReleaseRange, grabbableLayer) && 
// //                     !grabbableObject.transform.IsChildOf(craneHook) &&
// //                     !Physics.Raycast(ray, out hit, Mathf.Infinity, unloadingAreaLayer))
// //             {
// //                 Debug.Log("HERE in the Loading SignalPersonState");

// //                 currentState = SignalPersonState.Loading;
// //             }

// //             // Check if the grabbable object is attached to the crane hook and it is "BELOW" the obstacle
// //             else if (grabbableObject.transform.IsChildOf(craneHook) && 
// //                     craneHookHeight < obstacleBoundary &&
// //                     !Physics.Raycast(ray, out hit, Mathf.Infinity, firstFloorUnloadingBuildingLayer))
// //             {
// //                 currentState = SignalPersonState.HoistingCableUnloading;
// //             }

// //             // Check if the grabbable object is attached to the crane hook and it is "ABOVE" the obstacle
// //             else if (Mathf.Abs(signedAngularDistanceBwObjUnloading) > angularDistanceLimit &&
// //                     grabbableObject.transform.IsChildOf(craneHook) && craneHookHeight > obstacleBoundary && 
// //                     !Physics.Raycast(ray, out hit, Mathf.Infinity, unloadingAreaLayer))
// //             {
// //                 currentState = SignalPersonState.SwingAlignmentUnloading;
// //             }

// //             else if (Mathf.Abs(signedAngularDistanceBwObjUnloading) < angularDistanceLimit &&
// //                     grabbableObject.transform.IsChildOf(craneHook) &&
// //                     Mathf.Abs(signedAngularDistanceBwObjUnloading) < 5f &&
// //                     !Physics.Raycast(ray, out hit, Mathf.Infinity, unloadingAreaLayer))
// //             {
// //                 currentState = SignalPersonState.TrolleyAlignmentUnloading;
// //             }

// //             else if (Mathf.Abs(signedAngularDistanceBwObjUnloading) < angularDistanceLimit &&
// //                     Physics.Raycast(ray, out hit, Mathf.Infinity, unloadingAreaLayer) &&
// //                     grabbableObject.transform.IsChildOf(craneHook))
// //             {
// //                 currentState = SignalPersonState.LoweringCableUnloading;
// //             }

// //             else if (Mathf.Abs(signedAngularDistanceBwObjUnloading) < angularDistanceLimit && 
// //                     Physics.Raycast(ray, out hit, grabReleaseRange, unloadingAreaLayer) && 
// //                     grabbableObject.transform.IsChildOf(craneHook) &&
// //                     !Physics.Raycast(ray, out hit, Mathf.Infinity, grabbableLayer))
// //             {
// //                 currentState = SignalPersonState.Unloading;
// //             }

// //             else if (Mathf.Abs(signedAngularDistanceBwObjUnloading) < angularDistanceLimit &&
// //                     Physics.Raycast(ray, out hit, Mathf.Infinity, firstFloorUnloadingBuildingLayer) &&
// //                     !grabbableObject.transform.IsChildOf(craneHook) &&
// //                     !stopAfterCableDownForUnloadingSaid)
// //             {
// //                 currentState = SignalPersonState.Finish;
// //             }
            
// //             return currentState;
// //         }
// //     }


// /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// // Second approach: Using State Machine to simulate the signal person

// // using System.Collections;
// // using UnityEngine;

// // public enum SignalPersonGeneralState
// // {
// //     SwingAlignmentLoading,
// //     TrolleyAlignmentLoading,
// //     LoweringCableLoading,
// //     // Loading,
// //     HoistingCableUnloading,
// //     SwingAlignmentUnloading,
// //     TrolleyAlignmentUnloading,
// //     LoweringCableUnloading,
// //     Unloading,
// //     Finish,
// // }

// // public enum SignalPersonState
// // {
// //     SwingAlignmentLoading50ft,
// //     SwingAlignmentLoading30ft,
// //     SwingAlignmentLoading20ft,
// //     SwingAlignmentLoading10ft,
// //     SwingAlignmentLoading5ft,
// //     SwingAlignmentLoading2ft,

// //     TrolleyAlignmentLoading50ft,
// //     TrolleyAlignmentLoading30ft,
// //     TrolleyAlignmentLoading20ft,
// //     TrolleyAlignmentLoading10ft,
// //     TrolleyAlignmentLoading5ft,
// //     TrolleyAlignmentLoading2ft,

// //     LoweringCableLoading50ft,
// //     LoweringCableLoading30ft,
// //     LoweringCableLoading20ft,
// //     LoweringCableLoading10ft,
// //     LoweringCableLoading5ft,
// //     LoweringCableLoading2ft,
  
// //     // Loading,

// //     HoistingCableUnloading50ft,
// //     HoistingCableUnloading30ft,
// //     HoistingCableUnloading20ft,
// //     HoistingCableUnloading10ft,
// //     HoistingCableUnloading5ft,
// //     HoistingCableUnloading2ft,
    
// //     SwingAlignmentUnloading50ft,
// //     SwingAlignmentUnloading30ft,
// //     SwingAlignmentUnloading20ft,
// //     SwingAlignmentUnloading10ft,
// //     SwingAlignmentUnloading5ft,
// //     SwingAlignmentUnloading2ft,

// //     TrolleyAlignmentUnloading50ft,
// //     TrolleyAlignmentUnloading30ft,
// //     TrolleyAlignmentUnloading20ft,
// //     TrolleyAlignmentUnloading10ft,
// //     TrolleyAlignmentUnloading5ft,
// //     TrolleyAlignmentUnloading2ft,

// //     LoweringCableUnloading50ft,
// //     LoweringCableUnloading30ft,
// //     LoweringCableUnloading20ft,
// //     LoweringCableUnloading10ft,
// //     LoweringCableUnloading5ft,
// //     LoweringCableUnloading2ft,

// //     Unloading,

// //     Finish,
// // }

// // public class SignalPerson : MonoBehaviour
// // {
// //     // Define a delegate for callback function
// //     private delegate void LoadingCallback();

// //     private Transform grabbableObject;
// //     private Transform unloadingArea;
// //     private Transform obstacle;
// //     private LayerMask grabbableLayer;
// //     private LayerMask unloadingAreaLayer;
// //     private LayerMask obstacleLayer;
// //     private LayerMask firstFloorUnloadingBuildingLayer;
// //     [SerializeField] private AudioSource audioSource;

// //     private Transform craneHook;
// //     private Vector2 craneHookPos;
// //     private Vector2 grabbableObjectPos;
// //     private Vector2 unloadingAreaPos;
// //     private Vector3 cameraPosition;
// //     private Vector3 cameraPosOnXZ;
// //     private float signedAngularDistanceBwObjHook;
// //     private float signedAngularDistanceBwObjUnloading;
// //     private Ray ray;
// //     private RaycastHit hit;
// //     private float commandRepeatingTime = 15f;
// //     private float craneHookHeight;
// //     private float angularDistanceLimit = 5f;
// //     private float grabReleaseRange = 5f;
// //     private float obstacleMaximumHeightBoundary = 20f;
// //     private float obstacleBoundary;
// //     private bool stopAfterCableDownForUnloadingSaid = false;
// //     private bool stopAlreadySaid = false;

// //     private float distanceToCraneHook;
// //     private float distanceToUnloadingArea;
// //     private float distanceToGrabbableObject;
// //     private float horizontalDistanceBwHookObj;
// //     private float horizontalDistanceBwHookUnl;
// //     private float distanceAboveGrabbableObject;
// //     private float distanceAboveObstacle;
// //     private float distanceAboveUnloadingArea;

// //     private SignalPersonState currentState;
// //     private SignalPersonState previousState = SignalPersonState.Finish; // Initialize previousState to null
// //     private float lastCommandTime;

// //     // Define the thresholds for different angular distance commands
// //     float[] angularDistanceThresholds = { 50f, 30f, 20f, 10f, 5f, 2f };
// //     // Define the thresholds for different trolley movement commands
// //     float[] trolleyMovementThresholds = { 50f, 30f, 20f, 10f, 5f, 2f };
// //     // Define the thresholds for different cable distance commands
// //     float[] cableDistanceThresholds = { 50f, 30f, 20f, 10f, 5f, 2f };

// //     void Start()
// //     {
// //         craneHook = transform;
// //         grabbableLayer = 1 << LayerMask.NameToLayer("GrabbableObject");
// //         unloadingAreaLayer = 1 << LayerMask.NameToLayer("UnloadingArea");
// //         obstacleLayer = 1 << LayerMask.NameToLayer("Obstacle");
// //         firstFloorUnloadingBuildingLayer = 1 << LayerMask.NameToLayer("firstFloorUnloadingBuilding");

// //         // Find the grabbableObject in the scene based on its layer
// //         Collider[] grabbableColliders = Physics.OverlapSphere(Vector3.zero, Mathf.Infinity, grabbableLayer);
// //         grabbableObject = grabbableColliders[0].transform;

// //         // Find the unloadingArea in the scene based on its layer
// //         Collider[] unloadingColliders = Physics.OverlapSphere(Vector3.zero, Mathf.Infinity, unloadingAreaLayer);
// //         unloadingArea = unloadingColliders[0].transform;

// //         // Find the Obstacle in the scene based on its layer
// //         Collider[] obstacleColliders = Physics.OverlapSphere(Vector3.zero, Mathf.Infinity, obstacleLayer);
// //         obstacle = obstacleColliders[0].transform;
// //         // Raise the cable until the crane hook is obstacleMaximumHeightBoundary meters above the obstacle along the Y-axis
// //         Bounds obstacleBounds = new Bounds(obstacle.position, obstacle.localScale);
// //         obstacleBoundary = obstacleBounds.max.y + obstacleMaximumHeightBoundary;
    
// //         unloadingAreaPos = new Vector2(unloadingArea.position.x, unloadingArea.position.z);
// //     }

// //     // The following executes the actions for different Signal Person states in each frame
// //     void Update()
// //     {
// //         craneHookPos = new Vector2(craneHook.position.x, craneHook.position.z);
// //         grabbableObjectPos = new Vector2(grabbableObject.position.x, grabbableObject.position.z);

// //         // Measure angular distance between crane's hook and grabbable object's initial position
// //         signedAngularDistanceBwObjHook = Vector2.SignedAngle(craneHookPos, grabbableObjectPos)*2; // *2 is for adjusting- making angularDistance sound more real
// //         // Measure angular distance between crane's hook and unloading area's position
// //         signedAngularDistanceBwObjUnloading = Vector2.SignedAngle(craneHookPos, unloadingAreaPos);

// //         // Project Camera's position into X-Z plane
// //         cameraPosition = Camera.main.transform.position;
// //         cameraPosOnXZ = new Vector3(cameraPosition.x, cameraPosition.z);
// //         // Calculate distances in the X-Z plane
// //         distanceToCraneHook = Vector3.Distance(craneHookPos, cameraPosOnXZ);
// //         distanceToUnloadingArea = Vector3.Distance(unloadingAreaPos, cameraPosOnXZ);
// //         distanceToGrabbableObject = Vector3.Distance(grabbableObjectPos, cameraPosOnXZ);

// //         horizontalDistanceBwHookObj = distanceToGrabbableObject - distanceToCraneHook;
// //         horizontalDistanceBwHookUnl = distanceToUnloadingArea - distanceToCraneHook;

// //         // Draw a ray to see whether we are above the grabbable object
// //         ray = new Ray(transform.position, -transform.right);
// //         // Measuring crane hook height to compare it with obstacle's height
// //         craneHookHeight = craneHook.position.y;

// //         distanceAboveGrabbableObject = Mathf.Abs(craneHookHeight - grabbableObject.position.y);
// //         // Report the crane hook's distance above the obstacle
// //         distanceAboveObstacle = Mathf.Abs(craneHookHeight - obstacleBoundary);
// //         // Calculate distance above unloading area for "Cable Down" during unloading
// //         distanceAboveUnloadingArea = Mathf.Abs(craneHook.position.y - unloadingArea.position.y);

// //         // Call CurrentStateDesignator() function to get Signal Person's currentState
// //         currentState = CurrentStateDesignator();

// //         // Check if the general state has changed - Checking "Stop" command
// //         IsGeneralStateChanged(previousState, currentState);

// //         // Check if the current state is different from the previous one
// //         if (currentState != previousState)
// //         {
// //             // Send the command immediately when the state changes
// //             SendCommand();
// //         }
// //         // The following is for if currentState == previousState 
// //         else if (ShouldSendCommand())
// //         {
// //             // Send the command if it's time and the state hasn't changed
// //             SendCommand();
// //         }
// //     }

// //     // Helper function to check if the general state has changed
// //     // This function will handle "Stop" command when general state is changed
// //     private void IsGeneralStateChanged(SignalPersonState previousState, SignalPersonState newState)
// //     {
// //         // Check if the previous state and new state belong to different general states
// //         if (GetGeneralState(previousState) != GetGeneralState(newState)){
// //             stopAlreadySaid = false;
// //         }
// //         else{
// //             stopAlreadySaid = true;
// //         }
// //     }

// //     // Helper function to determine the general state
// //     SignalPersonGeneralState GetGeneralState(SignalPersonState state)
// //     {
// //         switch (state)
// //         {
// //             case SignalPersonState.SwingAlignmentLoading50ft:
// //             case SignalPersonState.SwingAlignmentLoading30ft:
// //             case SignalPersonState.SwingAlignmentLoading20ft:
// //             case SignalPersonState.SwingAlignmentLoading10ft:
// //             case SignalPersonState.SwingAlignmentLoading5ft:
// //             case SignalPersonState.SwingAlignmentLoading2ft:
// //                 return SignalPersonGeneralState.SwingAlignmentLoading;

// //             case SignalPersonState.TrolleyAlignmentLoading50ft:
// //             case SignalPersonState.TrolleyAlignmentLoading30ft:
// //             case SignalPersonState.TrolleyAlignmentLoading20ft:
// //             case SignalPersonState.TrolleyAlignmentLoading10ft:
// //             case SignalPersonState.TrolleyAlignmentLoading5ft:
// //             case SignalPersonState.TrolleyAlignmentLoading2ft:
// //                 return SignalPersonGeneralState.TrolleyAlignmentLoading;

// //             case SignalPersonState.LoweringCableLoading50ft:
// //             case SignalPersonState.LoweringCableLoading30ft:
// //             case SignalPersonState.LoweringCableLoading20ft:
// //             case SignalPersonState.LoweringCableLoading10ft:
// //             case SignalPersonState.LoweringCableLoading5ft:
// //             case SignalPersonState.LoweringCableLoading2ft:
// //                 return SignalPersonGeneralState.LoweringCableLoading;
        
// //             // case SignalPersonState.Loading:
// //             //     return SignalPersonGeneralState.Loading;

// //             case SignalPersonState.HoistingCableUnloading50ft:
// //             case SignalPersonState.HoistingCableUnloading30ft:
// //             case SignalPersonState.HoistingCableUnloading20ft:
// //             case SignalPersonState.HoistingCableUnloading10ft:
// //             case SignalPersonState.HoistingCableUnloading5ft:
// //             case SignalPersonState.HoistingCableUnloading2ft:
// //                 return SignalPersonGeneralState.HoistingCableUnloading;
            
// //             case SignalPersonState.SwingAlignmentUnloading50ft:
// //             case SignalPersonState.SwingAlignmentUnloading30ft:
// //             case SignalPersonState.SwingAlignmentUnloading20ft:
// //             case SignalPersonState.SwingAlignmentUnloading10ft:
// //             case SignalPersonState.SwingAlignmentUnloading5ft:
// //             case SignalPersonState.SwingAlignmentUnloading2ft:
// //                 return SignalPersonGeneralState.SwingAlignmentUnloading;

// //             case SignalPersonState.TrolleyAlignmentUnloading50ft:
// //             case SignalPersonState.TrolleyAlignmentUnloading30ft:
// //             case SignalPersonState.TrolleyAlignmentUnloading20ft:
// //             case SignalPersonState.TrolleyAlignmentUnloading10ft:
// //             case SignalPersonState.TrolleyAlignmentUnloading5ft:
// //             case SignalPersonState.TrolleyAlignmentUnloading2ft:
// //                 return SignalPersonGeneralState.TrolleyAlignmentUnloading;

// //             case SignalPersonState.LoweringCableUnloading50ft:
// //             case SignalPersonState.LoweringCableUnloading30ft:
// //             case SignalPersonState.LoweringCableUnloading20ft:
// //             case SignalPersonState.LoweringCableUnloading10ft:
// //             case SignalPersonState.LoweringCableUnloading5ft:
// //             case SignalPersonState.LoweringCableUnloading2ft:
// //                 return SignalPersonGeneralState.LoweringCableUnloading;

// //             case SignalPersonState.Unloading:
// //                 return SignalPersonGeneralState.Unloading;
            
// //             case SignalPersonState.Finish:
// //                 return SignalPersonGeneralState.Finish;


// //             default:
// //                 // If the state is not part of any known general state, return the state itself
// //                 return SignalPersonGeneralState.Finish;
// //         }
// //     }

// //     // Check if it's time to send a command
// //     bool ShouldSendCommand()
// //     {
// //         return Time.time - lastCommandTime >= commandRepeatingTime;
// //     }

// //     // Send the command based on the current state
// //     void SendCommand()
// //     {
// //         switch (currentState)
// //         {
// //             case SignalPersonState.SwingAlignmentLoading50ft:
// //             case SignalPersonState.SwingAlignmentLoading30ft:
// //             case SignalPersonState.SwingAlignmentLoading20ft:
// //             case SignalPersonState.SwingAlignmentLoading10ft:
// //             case SignalPersonState.SwingAlignmentLoading5ft:
// //             case SignalPersonState.SwingAlignmentLoading2ft:
// //                 SwingLoading();
// //                 break;

// //             case SignalPersonState.TrolleyAlignmentLoading50ft:
// //             case SignalPersonState.TrolleyAlignmentLoading30ft:
// //             case SignalPersonState.TrolleyAlignmentLoading20ft:
// //             case SignalPersonState.TrolleyAlignmentLoading10ft:
// //             case SignalPersonState.TrolleyAlignmentLoading5ft:
// //             case SignalPersonState.TrolleyAlignmentLoading2ft:
// //                 TrolleyLoading();
// //                 break;

// //             case SignalPersonState.LoweringCableLoading50ft:
// //             case SignalPersonState.LoweringCableLoading30ft:
// //             case SignalPersonState.LoweringCableLoading20ft:
// //             case SignalPersonState.LoweringCableLoading10ft:
// //             case SignalPersonState.LoweringCableLoading5ft:
// //             case SignalPersonState.LoweringCableLoading2ft:
// //                 CableDownLoading();
// //                 break;

// //             // case SignalPersonState.Loading:
// //             //     LoadingGrabbableObject();
// //             //     break;

// //             case SignalPersonState.HoistingCableUnloading50ft:
// //             case SignalPersonState.HoistingCableUnloading30ft:
// //             case SignalPersonState.HoistingCableUnloading20ft:
// //             case SignalPersonState.HoistingCableUnloading10ft:
// //             case SignalPersonState.HoistingCableUnloading5ft:
// //             case SignalPersonState.HoistingCableUnloading2ft:
// //                 CableUpUnloading();
// //                 break;

// //             case SignalPersonState.SwingAlignmentUnloading50ft:
// //             case SignalPersonState.SwingAlignmentUnloading30ft:
// //             case SignalPersonState.SwingAlignmentUnloading20ft:
// //             case SignalPersonState.SwingAlignmentUnloading10ft:
// //             case SignalPersonState.SwingAlignmentUnloading5ft:
// //             case SignalPersonState.SwingAlignmentUnloading2ft:
// //                 SwingUnloading();
// //                 break;

// //             case SignalPersonState.TrolleyAlignmentUnloading50ft:
// //             case SignalPersonState.TrolleyAlignmentUnloading30ft:
// //             case SignalPersonState.TrolleyAlignmentUnloading20ft:
// //             case SignalPersonState.TrolleyAlignmentUnloading10ft:
// //             case SignalPersonState.TrolleyAlignmentUnloading5ft:
// //             case SignalPersonState.TrolleyAlignmentUnloading2ft:
// //                 TrolleyUnloading();
// //                 break;

// //             case SignalPersonState.LoweringCableUnloading50ft:
// //             case SignalPersonState.LoweringCableUnloading30ft:
// //             case SignalPersonState.LoweringCableUnloading20ft:
// //             case SignalPersonState.LoweringCableUnloading10ft:
// //             case SignalPersonState.LoweringCableUnloading5ft:
// //             case SignalPersonState.LoweringCableUnloading2ft:
// //                 CableDownUnloading();
// //                 break;

// //             case SignalPersonState.Unloading:
// //                 LoadingGrabbableObject();
// //                 break;

// //             case SignalPersonState.Finish:
// //                 // Write the function for finishing unloading here
// //                 break;
// //         }

// //         // Update the time of the last command
// //         lastCommandTime = Time.time;
// //     }

// //     void SwingLoading()
// //     {
// //         // Check if any audio is currently playing
// //         if (audioSource.isPlaying)
// //         {
// //             // Stop any currently playing audio
// //             audioSource.Stop();
// //         }
// //         string angularDistanceCommand = GetAngularDistanceCommand(signedAngularDistanceBwObjHook, currentState);
// //         audioSource.PlayOneShot(GetAudioClip(angularDistanceCommand));
// //         previousState = currentState;
// //     }

// //     void SwingUnloading()
// //     {
// //         // Play stop when the Signal Person state is changed
// //         StartCoroutine(PlayStop(() =>
// //         {
// //             if (audioSource.isPlaying)
// //             {
// //                 audioSource.Stop();
// //             }

// //             string angularDistanceCommand = GetAngularDistanceCommand(signedAngularDistanceBwObjUnloading, currentState);
// //             audioSource.PlayOneShot(GetAudioClip(angularDistanceCommand));
// //         }));
// //         previousState = currentState;
// //     }

// //     private string GetAngularDistanceCommand(float distance, SignalPersonState state)
// //     {
// //             string stateName = state.ToString(); // Get the name of the enum value
// //             // Extract the numeric part of the enum name (e.g., "50", "30", etc.)
// //             string distanceString = stateName.Substring(stateName.IndexOf("ft") - 2, 2);
// //             // Parse the numeric part to get the distance value

// //             if (distance < 0)
// //             {
// //                 // Swing Right
// //                 // Iterate through the thresholds and return the first one that is greater than the distance
// //                 return "SwingRight" + distanceString + "ft";
// //             }
// //             else if (distance > 0)
// //             {
// //                 // Swing Right
// //                 // Iterate through the thresholds and return the first one that is greater than the distance
// //                 return "SwingLeft" + distanceString + "ft";
// //             }
// //         // If none of the thresholds are met, return nothing
// //         return null;
// //     }

// //     void TrolleyLoading()
// //     {
// //         // Play stop when the Signal Person state is changed
// //         StartCoroutine(PlayStop(() =>
// //         {
// //             if (audioSource.isPlaying)
// //             {
// //                 audioSource.Stop();
// //             }

// //             // Get the trolley movement command based on the distance
// //             string trolleyMovementCommand = GetTrolleyMovementCommand(distanceToCraneHook, distanceToGrabbableObject, currentState);
// //             audioSource.PlayOneShot(GetAudioClip(trolleyMovementCommand));
// //         }));
// //         previousState = currentState;
// //     }

// //     void TrolleyUnloading()
// //     {
// //         // Play stop when the Signal Person state is changed
// //         StartCoroutine(PlayStop(() =>
// //         {
// //             if (audioSource.isPlaying)
// //             {
// //                 audioSource.Stop();
// //             }

// //             // Get the trolley movement command based on the distance
// //             string trolleyMovementCommand = GetTrolleyMovementCommand(distanceToCraneHook, distanceToUnloadingArea, currentState);
// //             audioSource.PlayOneShot(GetAudioClip(trolleyMovementCommand));
// //         }));
// //         previousState = currentState;
// //     }

// //     private string GetTrolleyMovementCommand(float distanceToCraneHook, float distanceToObjectOrUnloading, SignalPersonState state)
// //     {
// //         string stateName = state.ToString(); // Get the name of the enum value
// //         string distanceString = stateName.Substring(stateName.IndexOf("ft") - 2, 2);
// //         // Parse the numeric part to get the distance value

// //         if (distanceToCraneHook <= distanceToObjectOrUnloading)
// //         {
// //             // Trolley Out
// //             return "TrolleyOut" + distanceString + "ft";
// //         }
// //         // Grabbable object is behind the crane's hook, move trolley in
// //         else if (distanceToCraneHook > distanceToObjectOrUnloading)
// //         {
// //             // Trolley In
// //             return "TrolleyIn" + distanceString + "ft";
// //         }
// //         return null;
// //     }

// //     void CableDownLoading()
// //     {
// //         StartCoroutine(PlayStop(() =>
// //         {
// //             if (audioSource.isPlaying)
// //             {
// //                 audioSource.Stop();
// //             }

// //             // Lower the cable until it is within 5 meters of the ground
// //             string cableDistanceCommand = GetCableDownDistanceCommand(distanceAboveGrabbableObject, currentState);
// //             audioSource.PlayOneShot(GetAudioClip(cableDistanceCommand));
// //         }));
// //         previousState = currentState;
// //     }

// //     void CableUpUnloading()
// //     {
// //         StartCoroutine(PlayStop(() =>
// //         {
// //             if (audioSource.isPlaying)
// //             {
// //                 audioSource.Stop();
// //             }

// //             string cableDistanceCommand = GetCableUpDistanceCommand(distanceAboveObstacle, currentState);
// //             audioSource.PlayOneShot(GetAudioClip(cableDistanceCommand));
// //         }));
// //         previousState = currentState;
// //     }

// //     void CableDownUnloading()
// //     {
// //         StartCoroutine(PlayStop(() =>
// //         {
// //             if (audioSource.isPlaying)
// //             {
// //                 audioSource.Stop();
// //             }

// //             // Lower the cable until it is within 5 meters of the unloadingArea
// //             // Report the crane hook's distance above the unloadingArea
// //             string cableDistanceCommand = GetCableDownDistanceCommand(distanceAboveUnloadingArea, currentState);
// //             audioSource.PlayOneShot(GetAudioClip(cableDistanceCommand));
// //         }));
// //         previousState = currentState;
// //     }

// //     private string GetCableDownDistanceCommand(float distance, SignalPersonState state)
// //     {
// //         string stateName = state.ToString(); // Get the name of the enum value
// //         string distanceString = stateName.Substring(stateName.IndexOf("ft") - 2, 2);
// //         // Parse the numeric part to get the distance value
// //         return "CableDown" + distanceString + "ft";
// //     }

// //     private string GetCableUpDistanceCommand(float distance, SignalPersonState state)
// //     {
// //         string stateName = state.ToString(); // Get the name of the enum value
// //         string distanceString = stateName.Substring(stateName.IndexOf("ft") - 2, 2);
// //         // Parse the numeric part to get the distance value
// //         return "CableUp" + distanceString + "ft";
// //     }

// //     // You can also use the following function for saying Stop while the Unloading is being started
// //     void LoadingGrabbableObject()
// //     {
// //         StartCoroutine(PlayStop(() => { }));
// //         previousState = currentState;
// //     }

// //     IEnumerator PlayStop(LoadingCallback callback)
// //     {
// //         if (!stopAlreadySaid)
// //         {
// //             // Check if any audio is currently playing
// //             if (audioSource.isPlaying)
// //             {
// //                 // Stop any currently playing audio
// //                 audioSource.Stop();
// //             }

// //             AudioClip audioClip = GetAudioClip("Stop");
// //             audioSource.PlayOneShot(audioClip);
// //             yield return new WaitForSeconds(audioClip.length + 1f); // Wait for the audio clip to finish playing (+1 second)
// //         }

// //         // At the end of PlayStop coroutine, call the callback function
// //         callback.Invoke();
// //     }

// //     private AudioClip GetAudioClip(string command)
// //     {
// //         // Access and return audio clip based on the command from the "Audio" folder
// //         // Assumes audio files are named correctly (e.g., SwingRight.mp3)
// //         string path = "Audio/" + command;

// //         return Resources.Load<AudioClip>(path);
// //     }

// //     SignalPersonState CurrentStateDesignator()
// //     {
// //         // Check if alignment is needed
// //         if (Mathf.Abs(signedAngularDistanceBwObjHook) > angularDistanceLimit && 
// //             !Physics.Raycast(ray, out hit, Mathf.Infinity, grabbableLayer) &&
// //             !grabbableObject.transform.IsChildOf(craneHook))
// //         {
// //             float signedAngularDistanceBwObjHookAbs = Mathf.Abs(signedAngularDistanceBwObjHook);

// //             if (signedAngularDistanceBwObjHookAbs > angularDistanceThresholds[0]){
// //                 currentState = SignalPersonState.SwingAlignmentLoading50ft;
// //             }
// //             else if (signedAngularDistanceBwObjHookAbs > angularDistanceThresholds[1]){
// //                 currentState = SignalPersonState.SwingAlignmentLoading30ft;
// //             }
// //             else if (signedAngularDistanceBwObjHookAbs > angularDistanceThresholds[2]){
// //                 currentState = SignalPersonState.SwingAlignmentLoading20ft;
// //             }
// //             else if (signedAngularDistanceBwObjHookAbs > angularDistanceThresholds[3]){
// //                 currentState = SignalPersonState.SwingAlignmentLoading10ft;
// //             }
// //             else if (signedAngularDistanceBwObjHookAbs > angularDistanceThresholds[4]){
// //                 currentState = SignalPersonState.SwingAlignmentLoading5ft;
// //             }
// //             else if (signedAngularDistanceBwObjHookAbs > angularDistanceThresholds[5]){
// //                 currentState = SignalPersonState.SwingAlignmentLoading2ft;
// //             }
// //             else{
// //                 currentState = SignalPersonState.SwingAlignmentLoading2ft;
// //             }            
// //         } 

// //         else if (Mathf.Abs(signedAngularDistanceBwObjHook) < angularDistanceLimit && 
// //                 !Physics.Raycast(ray, out hit, Mathf.Infinity, grabbableLayer) &&
// //                 !grabbableObject.transform.IsChildOf(craneHook))
// //         {
// //             // Comparing distance between hook and (grabbableObject/unloadingArea) to change the current state 
// //             // immediately after command (50ft, 30ft, etc.) change
// //             horizontalDistanceBwHookObj = Mathf.Abs(horizontalDistanceBwHookObj);

// //             if (horizontalDistanceBwHookObj > trolleyMovementThresholds[0]){
// //                 currentState = SignalPersonState.TrolleyAlignmentLoading50ft;
// //             }
// //             else if (horizontalDistanceBwHookObj > trolleyMovementThresholds[1]){
// //                 currentState = SignalPersonState.TrolleyAlignmentLoading30ft;
// //             }
// //             else if (horizontalDistanceBwHookObj > trolleyMovementThresholds[2]){
// //                 currentState = SignalPersonState.TrolleyAlignmentLoading20ft;
// //             }
// //             else if (horizontalDistanceBwHookObj > trolleyMovementThresholds[3]){
// //                 currentState = SignalPersonState.TrolleyAlignmentLoading10ft;
// //             }
// //             else if (horizontalDistanceBwHookObj > trolleyMovementThresholds[4]){
// //                 currentState = SignalPersonState.TrolleyAlignmentLoading5ft;
// //             }
// //             // You can erase the following "else if"; both are "else"
// //             else if (horizontalDistanceBwHookObj > trolleyMovementThresholds[5]){
// //                 currentState = SignalPersonState.TrolleyAlignmentLoading2ft;
// //             }
// //             else{
// //                 currentState = SignalPersonState.TrolleyAlignmentLoading2ft;
// //             }   
// //         }

// //         else if (Mathf.Abs(signedAngularDistanceBwObjHook) < angularDistanceLimit && 
// //                 Physics.Raycast(ray, out hit, Mathf.Infinity, grabbableLayer) && 
// //                 !grabbableObject.transform.IsChildOf(craneHook) &&
// //                 !Physics.Raycast(ray, out hit, Mathf.Infinity, unloadingAreaLayer))
// //         {
// //             if (distanceAboveGrabbableObject > cableDistanceThresholds[0]){
// //                 currentState = SignalPersonState.LoweringCableLoading50ft;
// //             }
// //             else if (distanceAboveGrabbableObject > cableDistanceThresholds[1]){
// //                 currentState = SignalPersonState.LoweringCableLoading30ft;
// //             }
// //             else if (distanceAboveGrabbableObject > cableDistanceThresholds[2]){
// //                 currentState = SignalPersonState.LoweringCableLoading20ft;
// //             }
// //             else if (distanceAboveGrabbableObject > cableDistanceThresholds[3]){
// //                 currentState = SignalPersonState.LoweringCableLoading10ft;
// //             }
// //             else if (distanceAboveGrabbableObject > cableDistanceThresholds[4]){
// //                 currentState = SignalPersonState.LoweringCableLoading5ft;
// //             }
// //             // You can erase the following "else if"; both are "else"
// //             else if (distanceAboveGrabbableObject > cableDistanceThresholds[5]){
// //                 currentState = SignalPersonState.LoweringCableLoading2ft;
// //             }
// //             else{
// //                 currentState = SignalPersonState.LoweringCableLoading2ft;
// //             }   
// //         }

// //         // else if (Mathf.Abs(signedAngularDistanceBwObjHook) < angularDistanceLimit && 
// //         //         Physics.Raycast(ray, out hit, grabReleaseRange, grabbableLayer) && 
// //         //         !grabbableObject.transform.IsChildOf(craneHook) &&
// //         //         !Physics.Raycast(ray, out hit, Mathf.Infinity, unloadingAreaLayer))
// //         // {
// //         //     currentState = SignalPersonState.Loading;
// //         // }

// //         // Check if the grabbable object is attached to the crane hook and it is "BELOW" the obstacle
// //         else if (grabbableObject.transform.IsChildOf(craneHook) && 
// //                 craneHookHeight < obstacleBoundary &&
// //                 !Physics.Raycast(ray, out hit, Mathf.Infinity, firstFloorUnloadingBuildingLayer))
// //         {
// //             if (distanceAboveObstacle > cableDistanceThresholds[0]){
// //                 currentState = SignalPersonState.HoistingCableUnloading50ft;
// //             }
// //             else if (distanceAboveObstacle > cableDistanceThresholds[1]){
// //                 currentState = SignalPersonState.HoistingCableUnloading30ft;
// //             }
// //             else if (distanceAboveObstacle > cableDistanceThresholds[2]){
// //                 currentState = SignalPersonState.HoistingCableUnloading20ft;
// //             }
// //             else if (distanceAboveObstacle > cableDistanceThresholds[3]){
// //                 currentState = SignalPersonState.HoistingCableUnloading10ft;
// //             }
// //             else if (distanceAboveObstacle > cableDistanceThresholds[4]){
// //                 currentState = SignalPersonState.HoistingCableUnloading5ft;
// //             }
// //             // You can erase the following "else if"; both are "else"
// //             else if (distanceAboveObstacle > cableDistanceThresholds[5]){
// //                 currentState = SignalPersonState.HoistingCableUnloading2ft;
// //             }
// //             else{
// //                 currentState = SignalPersonState.HoistingCableUnloading2ft;
// //             }   
// //         }

// //         // Check if the grabbable object is attached to the crane hook and it is "ABOVE" the obstacle
// //         else if (Mathf.Abs(signedAngularDistanceBwObjUnloading) > angularDistanceLimit &&
// //                 grabbableObject.transform.IsChildOf(craneHook) && craneHookHeight > obstacleBoundary && 
// //                 !Physics.Raycast(ray, out hit, Mathf.Infinity, unloadingAreaLayer))
// //         {
// //             float signedAngularDistanceBwObjUnloadingAbs = Mathf.Abs(signedAngularDistanceBwObjUnloading);

// //             if (signedAngularDistanceBwObjUnloadingAbs > angularDistanceThresholds[0]){
// //                 currentState = SignalPersonState.SwingAlignmentUnloading50ft;
// //             }
// //             else if (signedAngularDistanceBwObjUnloadingAbs > angularDistanceThresholds[1]){
// //                 currentState = SignalPersonState.SwingAlignmentUnloading30ft;
// //             }
// //             else if (signedAngularDistanceBwObjUnloadingAbs > angularDistanceThresholds[2]){
// //                 currentState = SignalPersonState.SwingAlignmentUnloading20ft;
// //             }
// //             else if (signedAngularDistanceBwObjUnloadingAbs > angularDistanceThresholds[3]){
// //                 currentState = SignalPersonState.SwingAlignmentUnloading10ft;
// //             }
// //             else if (signedAngularDistanceBwObjUnloadingAbs > angularDistanceThresholds[4]){
// //                 currentState = SignalPersonState.SwingAlignmentUnloading5ft;
// //             }
// //             else if (signedAngularDistanceBwObjUnloadingAbs > angularDistanceThresholds[5]){
// //                 currentState = SignalPersonState.SwingAlignmentUnloading2ft;
// //             }
// //             else{
// //                 currentState = SignalPersonState.SwingAlignmentUnloading2ft;
// //             }   
// //         }

// //         else if (Mathf.Abs(signedAngularDistanceBwObjUnloading) < angularDistanceLimit &&
// //                 grabbableObject.transform.IsChildOf(craneHook) &&
// //                 Mathf.Abs(signedAngularDistanceBwObjUnloading) < 5f &&
// //                 !Physics.Raycast(ray, out hit, Mathf.Infinity, unloadingAreaLayer))
// //         {
// //             // Comparing distance between hook and (grabbableObject/unloadingArea) to change the current state 
// //             // immediately after command (50ft, 30ft, etc.) change
// //             horizontalDistanceBwHookUnl = Mathf.Abs(horizontalDistanceBwHookUnl);

// //             if (horizontalDistanceBwHookUnl > trolleyMovementThresholds[0]){
// //                 currentState = SignalPersonState.TrolleyAlignmentUnloading50ft;
// //             }
// //             else if (horizontalDistanceBwHookUnl > trolleyMovementThresholds[1]){
// //                 currentState = SignalPersonState.TrolleyAlignmentUnloading30ft;
// //             }
// //             else if (horizontalDistanceBwHookUnl > trolleyMovementThresholds[2]){
// //                 currentState = SignalPersonState.TrolleyAlignmentUnloading20ft;
// //             }
// //             else if (horizontalDistanceBwHookUnl > trolleyMovementThresholds[3]){
// //                 currentState = SignalPersonState.TrolleyAlignmentUnloading10ft;
// //             }
// //             else if (horizontalDistanceBwHookUnl > trolleyMovementThresholds[4]){
// //                 currentState = SignalPersonState.TrolleyAlignmentUnloading5ft;
// //             }
// //             // You can erase the following "else if"; both are "else"
// //             else if (horizontalDistanceBwHookUnl > trolleyMovementThresholds[5]){
// //                 currentState = SignalPersonState.TrolleyAlignmentUnloading2ft;
// //             }
// //             else{
// //                 currentState = SignalPersonState.TrolleyAlignmentUnloading2ft;
// //             } 
// //         }

// //         else if (Mathf.Abs(signedAngularDistanceBwObjUnloading) < angularDistanceLimit &&
// //                 Physics.Raycast(ray, out hit, Mathf.Infinity, unloadingAreaLayer) &&
// //                 grabbableObject.transform.IsChildOf(craneHook))
// //         {
// //             if (distanceAboveUnloadingArea > cableDistanceThresholds[0]){
// //                 currentState = SignalPersonState.LoweringCableLoading50ft;
// //             }
// //             else if (distanceAboveUnloadingArea > cableDistanceThresholds[1]){
// //                 currentState = SignalPersonState.LoweringCableLoading30ft;
// //             }
// //             else if (distanceAboveUnloadingArea > cableDistanceThresholds[2]){
// //                 currentState = SignalPersonState.LoweringCableLoading20ft;
// //             }
// //             else if (distanceAboveUnloadingArea > cableDistanceThresholds[3]){
// //                 currentState = SignalPersonState.LoweringCableLoading10ft;
// //             }
// //             else if (distanceAboveUnloadingArea > cableDistanceThresholds[4]){
// //                 currentState = SignalPersonState.LoweringCableLoading5ft;
// //             }
// //             // You can erase the following "else if"; both are "else"
// //             else if (distanceAboveUnloadingArea > cableDistanceThresholds[5]){
// //                 currentState = SignalPersonState.LoweringCableLoading2ft;
// //             }
// //             else{
// //                 currentState = SignalPersonState.LoweringCableLoading2ft;
// //             }              
// //         }

// //         else if (Mathf.Abs(signedAngularDistanceBwObjUnloading) < angularDistanceLimit && 
// //                 Physics.Raycast(ray, out hit, grabReleaseRange, unloadingAreaLayer) && 
// //                 grabbableObject.transform.IsChildOf(craneHook) &&
// //                 !Physics.Raycast(ray, out hit, Mathf.Infinity, grabbableLayer))
// //         {
// //             currentState = SignalPersonState.Unloading;
// //         }

// //         else if (Mathf.Abs(signedAngularDistanceBwObjUnloading) < angularDistanceLimit &&
// //                 Physics.Raycast(ray, out hit, Mathf.Infinity, firstFloorUnloadingBuildingLayer) &&
// //                 !grabbableObject.transform.IsChildOf(craneHook) &&
// //                 !stopAfterCableDownForUnloadingSaid)
// //         {
// //             currentState = SignalPersonState.Finish;
// //         }
            
// //         return currentState;
// //         }
// //     }