using System.Collections;
using UnityEngine;

public enum SignalPersonBlindPickGeneralState
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

public enum SignalPersonBlindPickState
{
    SwingAlignmentLoading50ft,
    SwingAlignmentLoading30ft,
    SwingAlignmentLoading20ft,
    SwingAlignmentLoading10ft,
    SwingAlignmentLoading5ft,
    SwingAlignmentLoading2ft,

    TrolleyAlignmentLoading60ft,
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

    TrolleyAlignmentUnloading60ft,
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

public class SignalPersonBlindPick : MonoBehaviour
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
    private float horizontalDistanceLimit = 2f;
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

    private SignalPersonBlindPickState currentState;
    private SignalPersonBlindPickState previousState = SignalPersonBlindPickState.Finish; // Initialize previousState to null
    private float lastCommandTime;

    // Define the thresholds for different angular distance commands
    float[] angularDistanceThresholdsLoading = { 50f, 30f, 20f, 10f, 5f, 2f };
    // Angular distance thresholds for unloading the object 
    float[] angularDistanceThresholdsUnloading = { 60f, 50f, 30f, 20f, 10f, 5f, 2f };
    // Define the thresholds for different trolley movement commands
    float[] trolleyMovementThresholds = { 60f, 50f, 30f, 20f, 10f, 5f, 2f };    
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
    private void IsGeneralStateChanged(SignalPersonBlindPickState previousState, SignalPersonBlindPickState newState)
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
    SignalPersonBlindPickGeneralState GetGeneralState(SignalPersonBlindPickState state)
    {
        switch (state)
        {
            case SignalPersonBlindPickState.SwingAlignmentLoading50ft:
            case SignalPersonBlindPickState.SwingAlignmentLoading30ft:
            case SignalPersonBlindPickState.SwingAlignmentLoading20ft:
            case SignalPersonBlindPickState.SwingAlignmentLoading10ft:
            case SignalPersonBlindPickState.SwingAlignmentLoading5ft:
            case SignalPersonBlindPickState.SwingAlignmentLoading2ft:
                return SignalPersonBlindPickGeneralState.SwingAlignmentLoading;

            case SignalPersonBlindPickState.TrolleyAlignmentLoading60ft:
            case SignalPersonBlindPickState.TrolleyAlignmentLoading50ft:
            case SignalPersonBlindPickState.TrolleyAlignmentLoading30ft:
            case SignalPersonBlindPickState.TrolleyAlignmentLoading20ft:
            case SignalPersonBlindPickState.TrolleyAlignmentLoading10ft:
            case SignalPersonBlindPickState.TrolleyAlignmentLoading5ft:
            case SignalPersonBlindPickState.TrolleyAlignmentLoading2ft:
                return SignalPersonBlindPickGeneralState.TrolleyAlignmentLoading;

            case SignalPersonBlindPickState.LoweringCableLoading50ft:
            case SignalPersonBlindPickState.LoweringCableLoading30ft:
            case SignalPersonBlindPickState.LoweringCableLoading20ft:
                return SignalPersonBlindPickGeneralState.LoweringCableLoading;

            case SignalPersonBlindPickState.LoweringCableLoading10ft:
            case SignalPersonBlindPickState.LoweringCableLoading5ft:
            case SignalPersonBlindPickState.LoweringCableLoading2ft:
            case SignalPersonBlindPickState.Loading:
                return SignalPersonBlindPickGeneralState.Loading;

            case SignalPersonBlindPickState.HoistingCableUnloading50ft:
            case SignalPersonBlindPickState.HoistingCableUnloading30ft:
            case SignalPersonBlindPickState.HoistingCableUnloading20ft:
            case SignalPersonBlindPickState.HoistingCableUnloading10ft:
            case SignalPersonBlindPickState.HoistingCableUnloading5ft:
            case SignalPersonBlindPickState.HoistingCableUnloading2ft:
                return SignalPersonBlindPickGeneralState.HoistingCableUnloading;
            
            case SignalPersonBlindPickState.SwingAlignmentUnloading60ft:
            case SignalPersonBlindPickState.SwingAlignmentUnloading50ft:
            case SignalPersonBlindPickState.SwingAlignmentUnloading30ft:
            case SignalPersonBlindPickState.SwingAlignmentUnloading20ft:
            case SignalPersonBlindPickState.SwingAlignmentUnloading10ft:
            case SignalPersonBlindPickState.SwingAlignmentUnloading5ft:
            case SignalPersonBlindPickState.SwingAlignmentUnloading2ft:
                return SignalPersonBlindPickGeneralState.SwingAlignmentUnloading;

            case SignalPersonBlindPickState.TrolleyAlignmentUnloading60ft:
            case SignalPersonBlindPickState.TrolleyAlignmentUnloading50ft:
            case SignalPersonBlindPickState.TrolleyAlignmentUnloading30ft:
            case SignalPersonBlindPickState.TrolleyAlignmentUnloading20ft:
            case SignalPersonBlindPickState.TrolleyAlignmentUnloading10ft:
            case SignalPersonBlindPickState.TrolleyAlignmentUnloading5ft:
            case SignalPersonBlindPickState.TrolleyAlignmentUnloading2ft:
                return SignalPersonBlindPickGeneralState.TrolleyAlignmentUnloading;

            case SignalPersonBlindPickState.LoweringCableUnloading50ft:
            case SignalPersonBlindPickState.LoweringCableUnloading30ft:
            case SignalPersonBlindPickState.LoweringCableUnloading20ft:
                return SignalPersonBlindPickGeneralState.LoweringCableUnloading;

            case SignalPersonBlindPickState.LoweringCableUnloading10ft:
            case SignalPersonBlindPickState.LoweringCableUnloading5ft:
            case SignalPersonBlindPickState.LoweringCableUnloading2ft:
            case SignalPersonBlindPickState.Unloading:
                return SignalPersonBlindPickGeneralState.Unloading;
            
            case SignalPersonBlindPickState.Finish:
                return SignalPersonBlindPickGeneralState.Finish;


            default:
                // If the state is not part of any known general state, return the state itself
                return SignalPersonBlindPickGeneralState.Finish;
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
            case SignalPersonBlindPickState.SwingAlignmentLoading50ft:
            case SignalPersonBlindPickState.SwingAlignmentLoading30ft:
            case SignalPersonBlindPickState.SwingAlignmentLoading20ft:
            case SignalPersonBlindPickState.SwingAlignmentLoading10ft:
            case SignalPersonBlindPickState.SwingAlignmentLoading5ft:
            case SignalPersonBlindPickState.SwingAlignmentLoading2ft:
                SwingLoading();
                break;

            case SignalPersonBlindPickState.TrolleyAlignmentLoading60ft:
            case SignalPersonBlindPickState.TrolleyAlignmentLoading50ft:
            case SignalPersonBlindPickState.TrolleyAlignmentLoading30ft:
            case SignalPersonBlindPickState.TrolleyAlignmentLoading20ft:
            case SignalPersonBlindPickState.TrolleyAlignmentLoading10ft:
            case SignalPersonBlindPickState.TrolleyAlignmentLoading5ft:
            case SignalPersonBlindPickState.TrolleyAlignmentLoading2ft:
                TrolleyLoading();
                break;

            case SignalPersonBlindPickState.LoweringCableLoading50ft:
            case SignalPersonBlindPickState.LoweringCableLoading30ft:
            case SignalPersonBlindPickState.LoweringCableLoading20ft:
            case SignalPersonBlindPickState.LoweringCableLoading10ft:
                CableDownLoading();
                break;

            case SignalPersonBlindPickState.LoweringCableLoading5ft:
            case SignalPersonBlindPickState.LoweringCableLoading2ft:
            case SignalPersonBlindPickState.Loading:
                LoadingGrabbableObject();
                break;

            case SignalPersonBlindPickState.HoistingCableUnloading50ft:
            case SignalPersonBlindPickState.HoistingCableUnloading30ft:
            case SignalPersonBlindPickState.HoistingCableUnloading20ft:
            case SignalPersonBlindPickState.HoistingCableUnloading10ft:
            case SignalPersonBlindPickState.HoistingCableUnloading5ft:
            case SignalPersonBlindPickState.HoistingCableUnloading2ft:
                CableUpUnloading();
                break;

            case SignalPersonBlindPickState.SwingAlignmentUnloading60ft:
            case SignalPersonBlindPickState.SwingAlignmentUnloading50ft:
            case SignalPersonBlindPickState.SwingAlignmentUnloading30ft:
            case SignalPersonBlindPickState.SwingAlignmentUnloading20ft:
            case SignalPersonBlindPickState.SwingAlignmentUnloading10ft:
            case SignalPersonBlindPickState.SwingAlignmentUnloading5ft:
            case SignalPersonBlindPickState.SwingAlignmentUnloading2ft:
                SwingUnloading();
                break;

            case SignalPersonBlindPickState.TrolleyAlignmentUnloading60ft: 
            case SignalPersonBlindPickState.TrolleyAlignmentUnloading50ft:
            case SignalPersonBlindPickState.TrolleyAlignmentUnloading30ft:
            case SignalPersonBlindPickState.TrolleyAlignmentUnloading20ft:
            case SignalPersonBlindPickState.TrolleyAlignmentUnloading10ft:
            case SignalPersonBlindPickState.TrolleyAlignmentUnloading5ft:
            case SignalPersonBlindPickState.TrolleyAlignmentUnloading2ft:
                TrolleyUnloading();
                break;

            case SignalPersonBlindPickState.LoweringCableUnloading50ft:
            case SignalPersonBlindPickState.LoweringCableUnloading30ft:
            case SignalPersonBlindPickState.LoweringCableUnloading20ft:
                CableDownUnloading();
                break;

            case SignalPersonBlindPickState.LoweringCableUnloading10ft:
            case SignalPersonBlindPickState.LoweringCableUnloading5ft:
            case SignalPersonBlindPickState.LoweringCableUnloading2ft:
            case SignalPersonBlindPickState.Unloading:
                UnloadingGrabbableObject();
                break;

            case SignalPersonBlindPickState.Finish:
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

    private string GetAngularDistanceCommand(float distance, SignalPersonBlindPickState state)
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

    private string GetTrolleyMovementCommand(float distanceToCraneHook, float distanceToObjectOrUnloading, SignalPersonBlindPickState state)
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

    private string GetCableDownDistanceCommand(float distance, SignalPersonBlindPickState state)
    {
        string stateName = state.ToString(); // Get the name of the enum value
        string distanceString = stateName.Substring(stateName.IndexOf("ft") - 2, 2);
        // Parse the numeric part to get the distance value
        return "CableDown" + distanceString + "ft";
    }

    private string GetCableUpDistanceCommand(float distance, SignalPersonBlindPickState state)
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

    SignalPersonBlindPickState CurrentStateDesignator()
    {
        if (Physics.Raycast(ray, out hit, Mathf.Infinity, firstFloorUnloadingBuildingLayer) &&
                !grabbableObject.transform.IsChildOf(craneHook) &&
                !stopAfterCableDownForUnloadingSaid)
        {
            currentState = SignalPersonBlindPickState.Finish;
        }

        // Check if alignment is needed
        else if (Mathf.Abs(signedAngularDistanceBwObjHook) > angularDistanceLimit && 
            !Physics.Raycast(ray, out hit, Mathf.Infinity, grabbableLayer) &&
            !grabbableObject.transform.IsChildOf(craneHook))
        {
            float signedAngularDistanceBwObjHookAbs = Mathf.Abs(signedAngularDistanceBwObjHook);

            if (signedAngularDistanceBwObjHookAbs > angularDistanceThresholdsLoading[0]){
                currentState = SignalPersonBlindPickState.SwingAlignmentLoading50ft;
            }
            else if (signedAngularDistanceBwObjHookAbs > angularDistanceThresholdsLoading[1]){
                currentState = SignalPersonBlindPickState.SwingAlignmentLoading30ft;
            }
            else if (signedAngularDistanceBwObjHookAbs > angularDistanceThresholdsLoading[2]){
                currentState = SignalPersonBlindPickState.SwingAlignmentLoading20ft;
            }
            else if (signedAngularDistanceBwObjHookAbs > angularDistanceThresholdsLoading[3]){
                currentState = SignalPersonBlindPickState.SwingAlignmentLoading10ft;
            }
            else if (signedAngularDistanceBwObjHookAbs > angularDistanceThresholdsLoading[4]){
                currentState = SignalPersonBlindPickState.SwingAlignmentLoading5ft;
            }
            else if (signedAngularDistanceBwObjHookAbs > angularDistanceThresholdsLoading[5]){
                currentState = SignalPersonBlindPickState.SwingAlignmentLoading2ft;
            }
            else{
                currentState = SignalPersonBlindPickState.SwingAlignmentLoading2ft;
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
                currentState = SignalPersonBlindPickState.TrolleyAlignmentLoading60ft;
            }

            else if (horizontalDistanceBwHookObj > trolleyMovementThresholds[1]){
                currentState = SignalPersonBlindPickState.TrolleyAlignmentLoading50ft;
            }
            else if (horizontalDistanceBwHookObj > trolleyMovementThresholds[2]){
                currentState = SignalPersonBlindPickState.TrolleyAlignmentLoading30ft;
            }
            else if (horizontalDistanceBwHookObj > trolleyMovementThresholds[3]){
                currentState = SignalPersonBlindPickState.TrolleyAlignmentLoading20ft;
            }
            else if (horizontalDistanceBwHookObj > trolleyMovementThresholds[4]){
                currentState = SignalPersonBlindPickState.TrolleyAlignmentLoading10ft;
            }
            else if (horizontalDistanceBwHookObj > trolleyMovementThresholds[5]){
                currentState = SignalPersonBlindPickState.TrolleyAlignmentLoading5ft;
            }
            // You can erase the following "else if"; both are "else"
            else if (horizontalDistanceBwHookObj > trolleyMovementThresholds[6]){
                currentState = SignalPersonBlindPickState.TrolleyAlignmentLoading2ft;
            }
            else{
                currentState = SignalPersonBlindPickState.TrolleyAlignmentLoading2ft;
            }
        }

        else if (Mathf.Abs(signedAngularDistanceBwObjHook) < angularDistanceLimit && 
                Physics.Raycast(ray, out hit, Mathf.Infinity, grabbableLayer) && 
                !grabbableObject.transform.IsChildOf(craneHook) &&
                !Physics.Raycast(ray, out hit, Mathf.Infinity, unloadingAreaLayer))
        {
            if (distanceAboveGrabbableObject > cableDistanceThresholds[0]){
                currentState = SignalPersonBlindPickState.LoweringCableLoading50ft;
            }
            else if (distanceAboveGrabbableObject > cableDistanceThresholds[1]){
                currentState = SignalPersonBlindPickState.LoweringCableLoading30ft;
            }
            else if (distanceAboveGrabbableObject > cableDistanceThresholds[2]){
                currentState = SignalPersonBlindPickState.LoweringCableLoading20ft;
            }
            else if (distanceAboveGrabbableObject > cableDistanceThresholds[3]){
                currentState = SignalPersonBlindPickState.Loading;
            }
            else if (distanceAboveGrabbableObject > cableDistanceThresholds[4]){
                currentState = SignalPersonBlindPickState.Loading;
            }
            // You can erase the following "else if"; both are "else"
            else if (distanceAboveGrabbableObject > cableDistanceThresholds[5]){
                currentState = SignalPersonBlindPickState.Loading;
            }
            else{
                currentState = SignalPersonBlindPickState.Loading;
            }   
        }

        // else if (Mathf.Abs(signedAngularDistanceBwObjHook) < angularDistanceLimit && 
        //         Physics.Raycast(ray, out hit, grabReleaseRange, grabbableLayer) && 
        //         !grabbableObject.transform.IsChildOf(craneHook) &&
        //         !Physics.Raycast(ray, out hit, Mathf.Infinity, unloadingAreaLayer))
        // {
        //     currentState = SignalPersonBlindPickState.Loading;
        // }

        // Check if the grabbable object is attached to the crane hook and it is "BELOW" the obstacle
        else if (grabbableObject.transform.IsChildOf(craneHook) && 
                craneHookHeight < obstacleBoundary &&
                !Physics.Raycast(ray, out hit, Mathf.Infinity, firstFloorUnloadingBuildingLayer))
        {
            if (distanceAboveObstacle > cableDistanceThresholds[0]){
                currentState = SignalPersonBlindPickState.HoistingCableUnloading50ft;
            }
            else if (distanceAboveObstacle > cableDistanceThresholds[1]){
                currentState = SignalPersonBlindPickState.HoistingCableUnloading30ft;
            }
            else if (distanceAboveObstacle > cableDistanceThresholds[2]){
                currentState = SignalPersonBlindPickState.HoistingCableUnloading20ft;
            }
            else if (distanceAboveObstacle > cableDistanceThresholds[3]){
                currentState = SignalPersonBlindPickState.HoistingCableUnloading10ft;
            }
            else if (distanceAboveObstacle > cableDistanceThresholds[4]){
                currentState = SignalPersonBlindPickState.HoistingCableUnloading5ft;
            }
            // You can erase the following "else if"; both are "else"
            else if (distanceAboveObstacle > cableDistanceThresholds[5]){
                currentState = SignalPersonBlindPickState.HoistingCableUnloading2ft;
            }
            else{
                currentState = SignalPersonBlindPickState.HoistingCableUnloading2ft;
            }
        }

        else if (Mathf.Abs(horizontalDistanceBwHookUnl) > horizontalDistanceLimit &&
                grabbableObject.transform.IsChildOf(craneHook) &&
                !Physics.Raycast(ray, out hit, Mathf.Infinity, unloadingAreaLayer))
        {
            // Comparing distance between hook and (grabbableObject/unloadingArea) to change the current state 
            // immediately after command (50ft, 30ft, etc.) change
            horizontalDistanceBwHookUnl = Mathf.Abs(horizontalDistanceBwHookUnl);

            if (horizontalDistanceBwHookUnl > trolleyMovementThresholds[0]){
                currentState = SignalPersonBlindPickState.TrolleyAlignmentUnloading60ft;
            }
            else if (horizontalDistanceBwHookUnl > trolleyMovementThresholds[1]){
                currentState = SignalPersonBlindPickState.TrolleyAlignmentUnloading50ft;
            }
            else if (horizontalDistanceBwHookUnl > trolleyMovementThresholds[2]){
                currentState = SignalPersonBlindPickState.TrolleyAlignmentUnloading30ft;
            }
            else if (horizontalDistanceBwHookUnl > trolleyMovementThresholds[3]){
                currentState = SignalPersonBlindPickState.TrolleyAlignmentUnloading20ft;
            }
            else if (horizontalDistanceBwHookUnl > trolleyMovementThresholds[4]){
                currentState = SignalPersonBlindPickState.TrolleyAlignmentUnloading10ft;
            }
            else if (horizontalDistanceBwHookUnl > trolleyMovementThresholds[5]){
                currentState = SignalPersonBlindPickState.TrolleyAlignmentUnloading5ft;
            }
            // You can erase the following "else if"; both are "else"
            else if (horizontalDistanceBwHookUnl > trolleyMovementThresholds[6]){
                currentState = SignalPersonBlindPickState.TrolleyAlignmentUnloading2ft;
            }
            else{
                currentState = SignalPersonBlindPickState.TrolleyAlignmentUnloading2ft;
            } 
        }

        // Check if the grabbable object is attached to the crane hook and it is "ABOVE" the obstacle
        else if (Mathf.Abs(signedAngularDistanceBwObjUnloading) > angularDistanceLimit &&
                grabbableObject.transform.IsChildOf(craneHook) && craneHookHeight > obstacleBoundary && 
                !Physics.Raycast(ray, out hit, Mathf.Infinity, unloadingAreaLayer))
        {
            float signedAngularDistanceBwObjUnloadingAbs = Mathf.Abs(signedAngularDistanceBwObjUnloading);

            if (signedAngularDistanceBwObjUnloadingAbs > angularDistanceThresholdsUnloading[0]){
                currentState = SignalPersonBlindPickState.SwingAlignmentUnloading60ft;
            }
            else if (signedAngularDistanceBwObjUnloadingAbs > angularDistanceThresholdsUnloading[1]){
                currentState = SignalPersonBlindPickState.SwingAlignmentUnloading50ft;
            }
            else if (signedAngularDistanceBwObjUnloadingAbs > angularDistanceThresholdsUnloading[2]){
                currentState = SignalPersonBlindPickState.SwingAlignmentUnloading30ft;
            }
            else if (signedAngularDistanceBwObjUnloadingAbs > angularDistanceThresholdsUnloading[3]){
                currentState = SignalPersonBlindPickState.SwingAlignmentUnloading20ft;
            }
            else if (signedAngularDistanceBwObjUnloadingAbs > angularDistanceThresholdsUnloading[4]){
                currentState = SignalPersonBlindPickState.SwingAlignmentUnloading10ft;
            }
            else if (signedAngularDistanceBwObjUnloadingAbs > angularDistanceThresholdsUnloading[5]){
                currentState = SignalPersonBlindPickState.SwingAlignmentUnloading5ft;
            }
            else if (signedAngularDistanceBwObjUnloadingAbs > angularDistanceThresholdsUnloading[6]){
                currentState = SignalPersonBlindPickState.SwingAlignmentUnloading2ft;
            }
            else{
                currentState = SignalPersonBlindPickState.SwingAlignmentUnloading2ft;
            }   
        }

        else if (Mathf.Abs(signedAngularDistanceBwObjUnloading) < angularDistanceLimit &&
                Physics.Raycast(ray, out hit, Mathf.Infinity, unloadingAreaLayer) &&
                grabbableObject.transform.IsChildOf(craneHook))
        {
            if (distanceAboveUnloadingArea > cableDistanceThresholds[0]){
                currentState = SignalPersonBlindPickState.LoweringCableLoading50ft;
            }
            else if (distanceAboveUnloadingArea > cableDistanceThresholds[1]){
                currentState = SignalPersonBlindPickState.LoweringCableLoading30ft;
            }
            else if (distanceAboveUnloadingArea > cableDistanceThresholds[2]){
                currentState = SignalPersonBlindPickState.LoweringCableLoading20ft;
            }
            else if (distanceAboveUnloadingArea > cableDistanceThresholds[3]){
                currentState = SignalPersonBlindPickState.Unloading;
            }
            else if (distanceAboveUnloadingArea > cableDistanceThresholds[4]){
                currentState = SignalPersonBlindPickState.Unloading;
            }
            // You can erase the following "else if"; both are "else"
            else if (distanceAboveUnloadingArea > cableDistanceThresholds[5]){
                currentState = SignalPersonBlindPickState.Unloading;
            }
            else{
                currentState = SignalPersonBlindPickState.Unloading;
            }              
        }

        else if (Mathf.Abs(signedAngularDistanceBwObjUnloading) < angularDistanceLimit && 
                Physics.Raycast(ray, out hit, grabReleaseRange, unloadingAreaLayer) && 
                grabbableObject.transform.IsChildOf(craneHook) &&
                !Physics.Raycast(ray, out hit, Mathf.Infinity, grabbableLayer))
        {
            currentState = SignalPersonBlindPickState.Unloading;
        }
            
        return currentState;
        }
    }