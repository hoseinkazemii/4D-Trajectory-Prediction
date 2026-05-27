using System.Collections;
using UnityEngine;

public class MoveObject : MonoBehaviour
{
    private LayerMask grabbableLayer;
    private LayerMask unloadingStoreyLayer;
    private LayerMask chainLinkedBelowTheRopeLayer;

    private float grabReleaseRange = 30f;
    private Transform grabbedObject;
    private bool objectReleased = false;
    private Transform lowestLink;

    public bool canGrab = true;
    public bool unloadingFinished = false;

    private ObjectMovement objectMovement;
    private float buttonHoldTime = 0;
    private bool isGrabbingButtonHeld = false;
    private bool isReleasingButtonHeld = false;

    private bool craneHookAboveGrabbableObject = false;
    private bool craneHookAboveUnloadingStorey = false;
    private AudioClip chainSoundEffect;
    public AudioSource chainAudioSource;

    public ProgressBarManager progressBarManager; // Assign this in the Inspector
    public ProgressBarManagerBlind progressBarManagerBlind; // Assign this in the Inspector

    public CabinPanelManager cabinPanelManager; // Assign this in the Inspector
    private float objectWeight = 1183.1f; // Example weight of the grabbable object
    private float currentWeight = 0f; // Current weight displayed on the panel
    private float loadingTime = 5f; // Total time to load the object

    void Awake()
    {
        objectMovement = new ObjectMovement();
        objectMovement.Enable(); // Enable the input action map

        // Subscribe to the performed and canceled events
        objectMovement.GrabOrReleaseObject.GrabObject.performed += ctx => StartGrabbingButtonHold();
        objectMovement.GrabOrReleaseObject.GrabObject.canceled += ctx => EndGrabbingButtonHold();
        objectMovement.GrabOrReleaseObject.ReleaseObject.performed += ctx => StartReleasingButtonHold();
        objectMovement.GrabOrReleaseObject.ReleaseObject.canceled += ctx => EndReleasingButtonHold();

        chainSoundEffect = Resources.Load<AudioClip>("Audio/ChainSoundEffect");
    }

    void Start()
    {
        grabbableLayer = 1 << LayerMask.NameToLayer("GrabbableObject");
        unloadingStoreyLayer = 1 << LayerMask.NameToLayer("UnloadingStorey");
        chainLinkedBelowTheRopeLayer = 1 << LayerMask.NameToLayer("linkBelowTheRope");
    }

    IEnumerator MoveObjectToHookPosition(Transform objectToAttach)
    {
        float grabbableObjectAttachingDuration = 2.0f; // Duration over which the object moves to the crane hook
        Vector3 startPosition = objectToAttach.position;
        Collider[] colliders = Physics.OverlapSphere(transform.position, 500f, chainLinkedBelowTheRopeLayer);
        if (colliders.Length > 0)
        {
            lowestLink = colliders[0].transform;
            Vector3 endPosition = lowestLink.position;

            float elapsedTime = 0;
            while (elapsedTime < grabbableObjectAttachingDuration)
            {
                objectToAttach.position = Vector3.Lerp(startPosition, endPosition, elapsedTime / grabbableObjectAttachingDuration);
                elapsedTime += Time.deltaTime;
                yield return null;
            }
            objectToAttach.position = endPosition;
            StartCoroutine(AttachAfterDelay(objectToAttach)); // Start attaching after moving
        }
    }

    IEnumerator AttachAfterDelay(Transform objectToAttach)
    {
        objectToAttach.SetParent(transform);

        // Create a joint and initially set it to be very 'soft'
        ConfigurableJoint joint = objectToAttach.gameObject.AddComponent<ConfigurableJoint>();
        joint.connectedBody = lowestLink.GetComponent<Rigidbody>();
        joint.xMotion = joint.yMotion = joint.zMotion = ConfigurableJointMotion.Limited;
        joint.angularXMotion = joint.angularYMotion = joint.angularZMotion = ConfigurableJointMotion.Limited;
        joint.autoConfigureConnectedAnchor = false;
        joint.connectedAnchor = Vector3.zero;

        JointDrive drive = new JointDrive { positionSpring = 0, positionDamper = 0, maximumForce = Mathf.Infinity };
        joint.xDrive = joint.yDrive = joint.zDrive = drive;

        float targetSpring = 100; // Target stiffness
        float targetDamper = 5; // Target damping
        float attachingObjectDuration = 5f;
        float elapsedTimeForAttachingObject = 0f;

        while (elapsedTimeForAttachingObject < attachingObjectDuration)
        {
            elapsedTimeForAttachingObject += Time.deltaTime;
            float t = elapsedTimeForAttachingObject / attachingObjectDuration;

            float spring = Mathf.Lerp(0, targetSpring, t);
            float damper = Mathf.Lerp(0, targetDamper, t);

            drive.positionSpring = spring;
            drive.positionDamper = damper;
            joint.xDrive = joint.yDrive = drive;

            yield return null;
        }
    }

    private void Update()
    {
        UpdateCranePositionFlags(); // Update flags at the start of each frame
        
        if (canGrab && !objectReleased)
        {
            Ray ray = new Ray(transform.position, -transform.right);
            RaycastHit hit;

            if (Physics.Raycast(ray, out hit, grabReleaseRange, grabbableLayer) && isGrabbingButtonHeld)
            {
                buttonHoldTime += Time.deltaTime;
                UpdateLoadWeight(buttonHoldTime / loadingTime * objectWeight); // Update load weight on panel

                if (buttonHoldTime >= 5f)
                {
                    grabbedObject = hit.collider.transform;
                    StartCoroutine(MoveObjectToHookPosition(grabbedObject));
                    canGrab = false; // Prevents grabbing in the same frame
                    buttonHoldTime = 0;
                    isGrabbingButtonHeld = false;
                }
            }
        }

        if (grabbedObject != null && !objectReleased)
        {
            RaycastHit hitFromAttached;

            if (Physics.Raycast(grabbedObject.position, Vector3.down, out hitFromAttached, grabReleaseRange, unloadingStoreyLayer) && isReleasingButtonHeld)
            {
                buttonHoldTime += Time.deltaTime;
                if (buttonHoldTime >= 5f)
                {
                    Destroy(grabbedObject.GetComponent<ConfigurableJoint>());
                    grabbedObject.SetParent(null);
                    grabbedObject = null;
                    objectReleased = true;
                    unloadingFinished = true;
                    buttonHoldTime = 0;
                    isGrabbingButtonHeld = false;

                    UpdateLoadWeight(0); // Reset load weight on panel
                }
            }
        }
    }

    private void UpdateCranePositionFlags()
    {
        Ray ray = new Ray(transform.position, -transform.right);
        RaycastHit hit;

        craneHookAboveGrabbableObject = canGrab && grabbedObject == null && Physics.Raycast(ray, out hit, grabReleaseRange, grabbableLayer);

        Ray rayDown = new Ray(grabbedObject?.position ?? Vector3.zero, Vector3.down);
        craneHookAboveUnloadingStorey = grabbedObject != null && Physics.Raycast(rayDown, out hit, grabReleaseRange, unloadingStoreyLayer);
    }

    private void UpdateLoadWeight(float weight)
    {
        currentWeight = weight;
        cabinPanelManager.SetLoadWeight(currentWeight);
    }

    private void StartGrabbingButtonHold()
    {
        isGrabbingButtonHeld = true;
        buttonHoldTime = 0; // Reset the timer when button press starts
        if (craneHookAboveGrabbableObject){
            progressBarManager.StartButtonHold(); // Start the progress bar
            progressBarManagerBlind.StartButtonHold();
            chainAudioSource.PlayOneShot(chainSoundEffect);
        }
    }

    private void EndGrabbingButtonHold()
    {
        isGrabbingButtonHeld = false;
        buttonHoldTime = 0; // Reset the timer when button is released
        progressBarManager.EndButtonHold(); // Hide the progress bar
        progressBarManagerBlind.EndButtonHold(); // Hide the progress bar
        chainAudioSource.Stop(); // Stop the chain sound when grabbing is cancelled or completed
    }

    private void StartReleasingButtonHold()
    {
        isReleasingButtonHeld = true;
        buttonHoldTime = 0; // Reset the timer when button press starts
        if (craneHookAboveUnloadingStorey){
            progressBarManager.StartButtonHold(); // Start the progress bar
            progressBarManagerBlind.StartButtonHold(); // Start the progress bar
            chainAudioSource.PlayOneShot(chainSoundEffect);
        }
    }

    private void EndReleasingButtonHold()
    {
        isReleasingButtonHeld = false;
        buttonHoldTime = 0; // Reset the timer when button is released
        progressBarManager.EndButtonHold(); // Hide the progress bar
        progressBarManagerBlind.EndButtonHold(); // Hide the progress bar
        chainAudioSource.Stop();
    }

    void OnDestroy()
    {
        // Unsubscribe to prevent memory leaks
        objectMovement.GrabOrReleaseObject.GrabObject.performed -= ctx => StartGrabbingButtonHold();
        objectMovement.GrabOrReleaseObject.GrabObject.canceled -= ctx => EndGrabbingButtonHold();
        objectMovement.GrabOrReleaseObject.GrabObject.performed -= ctx => StartReleasingButtonHold();
        objectMovement.GrabOrReleaseObject.GrabObject.canceled -= ctx => EndReleasingButtonHold();
        objectMovement.Disable(); // Disable the input action map
    }
}




















// using System.Collections;
// using UnityEngine;

// public class MoveObject : MonoBehaviour
// {
//     private LayerMask grabbableLayer;
//     // private LayerMask unloadingAreaLayer;
//     private LayerMask unloadingStoreyLayer;
//     private LayerMask chainLinkedBelowTheRopeLayer;

//     private float grabReleaseRange = 30f;
//     private Transform grabbedObject;
//     // private float loadingTime = 5f;
//     private bool objectReleased = false;
//     private Transform lowestLink; // Declare this at the class level

//     public bool canGrab = true;
//     public bool unloadingFinished = false;

//     private ObjectMovement objectMovement;
//     private float buttonHoldTime = 0;
//     private bool isGrabbingButtonHeld = false;
//     private bool isReleasingButtonHeld = false;

//     private bool craneHookAboveGrabbableObject = false;
//     private bool craneHookAboveUnloadingStorey = false;
//     private AudioClip chainSoundEffect;
//     public AudioSource chainAudioSource;

//     public ProgressBarManager progressBarManager; // Assign this in the Inspector
//     public ProgressBarManagerBlind progressBarManagerBlind; // Assign this in the Inspector

    
//     void Awake()
//     {
//         objectMovement = new ObjectMovement();
//         objectMovement.Enable(); // Enable the input action map

//         // Subscribe to the performed and canceled events
//         objectMovement.GrabOrReleaseObject.GrabObject.performed += ctx => StartGrabbingButtonHold();
//         objectMovement.GrabOrReleaseObject.GrabObject.canceled += ctx => EndGrabbingButtonHold();
//         objectMovement.GrabOrReleaseObject.ReleaseObject.performed += ctx => StartReleasingButtonHold();
//         objectMovement.GrabOrReleaseObject.ReleaseObject.canceled += ctx => EndReleasingButtonHold();

//         chainSoundEffect = Resources.Load<AudioClip>("Audio/ChainSoundEffect");
//     }


//     void Start()
//     {
//         grabbableLayer = 1 << LayerMask.NameToLayer("GrabbableObject");
//         // unloadingAreaLayer = 1 << LayerMask.NameToLayer("UnloadingArea");
//         unloadingStoreyLayer = 1 << LayerMask.NameToLayer("UnloadingStorey");
//         chainLinkedBelowTheRopeLayer = 1 << LayerMask.NameToLayer("linkBelowTheRope");
//     }

//     IEnumerator MoveObjectToHookPosition(Transform objectToAttach)
//     {
//         float grabbableObjectAttachingDuration = 2.0f; // Duration over which the object moves to the crane hook
//         Vector3 startPosition = objectToAttach.position;
//         Collider[] colliders = Physics.OverlapSphere(transform.position, 500f, chainLinkedBelowTheRopeLayer);
//         if (colliders.Length > 0)
//         {
//             lowestLink = colliders[0].transform;
//             Vector3 endPosition = lowestLink.position;

//             float elapsedTime = 0;
//             while (elapsedTime < grabbableObjectAttachingDuration)
//             {
//                 objectToAttach.position = Vector3.Lerp(startPosition, endPosition, elapsedTime / grabbableObjectAttachingDuration);
//                 elapsedTime += Time.deltaTime;
//                 yield return null;
//             }
//             objectToAttach.position = endPosition;
//             StartCoroutine(AttachAfterDelay(objectToAttach)); // Start attaching after moving
//         }
//     }

//     IEnumerator AttachAfterDelay(Transform objectToAttach)
//     {
//         // yield return new WaitForSeconds(loadingTime);
//         objectToAttach.SetParent(transform);

//         // Create a joint and initially set it to be very 'soft'
//         ConfigurableJoint joint = objectToAttach.gameObject.AddComponent<ConfigurableJoint>();
//         joint.connectedBody = lowestLink.GetComponent<Rigidbody>();
//         joint.xMotion = joint.yMotion = joint.zMotion = ConfigurableJointMotion.Limited;
//         joint.angularXMotion = joint.angularYMotion = joint.angularZMotion = ConfigurableJointMotion.Limited;
//         joint.autoConfigureConnectedAnchor = false;
//         joint.connectedAnchor = Vector3.zero;

//         JointDrive drive = new JointDrive { positionSpring = 0, positionDamper = 0, maximumForce = Mathf.Infinity };
//         joint.xDrive = joint.yDrive = joint.zDrive = drive;

//         float targetSpring = 100; // Target stiffness
//         float targetDamper = 5; // Target damping
//         float attachingObjectDuration = 5f;
//         float elapsedTimeForAttachingObject = 0f;

//         while (elapsedTimeForAttachingObject < attachingObjectDuration)
//         {
//             elapsedTimeForAttachingObject += Time.deltaTime;
//             float t = elapsedTimeForAttachingObject / attachingObjectDuration;

//             float spring = Mathf.Lerp(0, targetSpring, t);
//             float damper = Mathf.Lerp(0, targetDamper, t);

//             drive.positionSpring = spring;
//             drive.positionDamper = damper;
//             joint.xDrive = joint.yDrive = joint.zDrive = drive;

//             yield return null;
//         }
//     }

//     private void Update()
//     {
//         UpdateCranePositionFlags(); // Update flags at the start of each frame
        
//         if (canGrab && !objectReleased)
//         {
//             Ray ray = new Ray(transform.position, -transform.right);
//             RaycastHit hit;

//             if (Physics.Raycast(ray, out hit, grabReleaseRange, grabbableLayer) && isGrabbingButtonHeld)
//             {
//                 buttonHoldTime += Time.deltaTime;
//                 // Check if the button has been held for more than 2 seconds
//                 if (buttonHoldTime >= 5f)
//                 {
//                     grabbedObject = hit.collider.transform;
//                     StartCoroutine(MoveObjectToHookPosition(grabbedObject));
//                     canGrab = false; // Prevents grabbing in the same frame
//                     // Reset hold time to avoid multiple triggers
//                     buttonHoldTime = 0;
//                     isGrabbingButtonHeld = false;
//                 }
//             }
//         }

//         if (grabbedObject != null && !objectReleased)
//         {
//             // Check if the grabbed object is above the unloading area after attaching
//             RaycastHit hitFromAttached;

//             if (Physics.Raycast(grabbedObject.position, Vector3.down, out hitFromAttached, grabReleaseRange, unloadingStoreyLayer) && isReleasingButtonHeld)
//             {
//                 buttonHoldTime += Time.deltaTime;
//                 // Check if the button has been held for more than 2 seconds
//                 if (buttonHoldTime >= 5f)
//                 {
//                     // Release the object from the crane hook by removing the configurable joint
//                     Destroy(grabbedObject.GetComponent<ConfigurableJoint>());
//                     grabbedObject.SetParent(null);
//                     grabbedObject = null;
//                     objectReleased = true;
//                     // canGrab = true; // Allow grabbing in the next frame
//                     unloadingFinished = true;

//                     // Reset hold time to avoid multiple triggers
//                     buttonHoldTime = 0;
//                     isGrabbingButtonHeld = false;
//                 }
//             }
//         }
//     }

//     private void UpdateCranePositionFlags()
//     {
//         // Update craneHookAboveGrabbableObject and craneHookAboveUnloadingArea here based on your raycasting logic
//         Ray ray = new Ray(transform.position, -transform.right);
//         RaycastHit hit;

//         craneHookAboveGrabbableObject = canGrab && grabbedObject == null && Physics.Raycast(ray, out hit, grabReleaseRange, grabbableLayer);

//         Ray rayDown = new Ray(grabbedObject?.position ?? Vector3.zero, Vector3.down);
//         craneHookAboveUnloadingStorey = grabbedObject != null && Physics.Raycast(rayDown, out hit, grabReleaseRange, unloadingStoreyLayer);
//     }

//     private void StartGrabbingButtonHold()
//     {
//         isGrabbingButtonHeld = true;
//         buttonHoldTime = 0; // Reset the timer when button press starts
//         if (craneHookAboveGrabbableObject){
//             progressBarManager.StartButtonHold(); // Start the progress bar
//             progressBarManagerBlind.StartButtonHold();
//             chainAudioSource.PlayOneShot(chainSoundEffect);
//         }
//     }

//     private void EndGrabbingButtonHold()
//     {
//         isGrabbingButtonHeld = false;
//         buttonHoldTime = 0; // Reset the timer when button is released
//         progressBarManager.EndButtonHold(); // Hide the progress bar
//         progressBarManagerBlind.EndButtonHold(); // Hide the progress bar
//         chainAudioSource.Stop(); // Stop the chain sound when grabbing is cancelled or completed
//     }

//     private void StartReleasingButtonHold()
//     {
//         isReleasingButtonHeld = true;
//         buttonHoldTime = 0; // Reset the timer when button press starts
//         if (craneHookAboveUnloadingStorey){
//             progressBarManager.StartButtonHold(); // Start the progress bar
//             progressBarManagerBlind.StartButtonHold(); // Start the progress bar
//             chainAudioSource.PlayOneShot(chainSoundEffect);
//         }
//     }

//     private void EndReleasingButtonHold()
//     {
//         isReleasingButtonHeld = false;
//         buttonHoldTime = 0; // Reset the timer when button is released
//         progressBarManager.EndButtonHold(); // Hide the progress bar
//         progressBarManagerBlind.EndButtonHold(); // Hide the progress bar
//         chainAudioSource.Stop();
//     }

//     void OnDestroy()
//     {
//         // Unsubscribe to prevent memory leaks
//         objectMovement.GrabOrReleaseObject.GrabObject.performed -= ctx => StartGrabbingButtonHold();
//         objectMovement.GrabOrReleaseObject.GrabObject.canceled -= ctx => EndGrabbingButtonHold();
//         objectMovement.GrabOrReleaseObject.GrabObject.performed -= ctx => StartReleasingButtonHold();
//         objectMovement.GrabOrReleaseObject.GrabObject.canceled -= ctx => EndReleasingButtonHold();
//         objectMovement.Disable(); // Disable the input action map
//     }
// }


/////////////////////////////////////////////////////////////////////
//// Suddenly grabbing the object
// using System.Collections;
// using UnityEngine;

// public class MoveObject : MonoBehaviour
// {
//     private LayerMask grabbableLayer;
//     // private LayerMask unloadingAreaLayer;
//     private LayerMask unloadingStoreyLayer;
//     private LayerMask chainLinkedBelowTheRopeLayer;

//     private float grabReleaseRange = 30f;
//     private Transform grabbedObject;
//     // private float loadingTime = 5f;
//     private bool objectReleased = false;

//     public bool canGrab = true;
//     public bool unloadingFinished = false;

//     private ObjectMovement objectMovement;
//     private float buttonHoldTime = 0;
//     private bool isGrabbingButtonHeld = false;
//     private bool isReleasingButtonHeld = false;

//     private bool craneHookAboveGrabbableObject = false;
//     private bool craneHookAboveUnloadingStorey = false;
//     private AudioClip chainSoundEffect;
//     public AudioSource chainAudioSource;

//     public ProgressBarManager progressBarManager; // Assign this in the Inspector
//     public ProgressBarManagerBlind progressBarManagerBlind; // Assign this in the Inspector

    
//     void Awake()
//     {
//         objectMovement = new ObjectMovement();
//         objectMovement.Enable(); // Enable the input action map

//         // Subscribe to the performed and canceled events
//         objectMovement.GrabOrReleaseObject.GrabObject.performed += ctx => StartGrabbingButtonHold();
//         objectMovement.GrabOrReleaseObject.GrabObject.canceled += ctx => EndGrabbingButtonHold();
//         objectMovement.GrabOrReleaseObject.ReleaseObject.performed += ctx => StartReleasingButtonHold();
//         objectMovement.GrabOrReleaseObject.ReleaseObject.canceled += ctx => EndReleasingButtonHold();

//         chainSoundEffect = Resources.Load<AudioClip>("Audio/ChainSoundEffect");
//     }


//     void Start()
//     {
//         grabbableLayer = 1 << LayerMask.NameToLayer("GrabbableObject");
//         // unloadingAreaLayer = 1 << LayerMask.NameToLayer("UnloadingArea");
//         unloadingStoreyLayer = 1 << LayerMask.NameToLayer("UnloadingStorey");
//         chainLinkedBelowTheRopeLayer = 1 << LayerMask.NameToLayer("linkBelowTheRope");
//     }

//     IEnumerator AttachAfterDelay(Transform objectToAttach)
//     {
//         // yield return new WaitForSeconds(loadingTime);
//         objectToAttach.SetParent(transform);

//         Collider[] colliders = Physics.OverlapSphere(transform.position, 500f, chainLinkedBelowTheRopeLayer);
//         if (colliders.Length > 0)
//         {
//             Transform lowestLink = colliders[0].transform;
//             objectToAttach.position = lowestLink.position; // Initially position object correctly

//             // Create a joint and initially set it to be very 'soft'
//             ConfigurableJoint joint = objectToAttach.gameObject.AddComponent<ConfigurableJoint>();
//             joint.connectedBody = lowestLink.GetComponent<Rigidbody>();
//             joint.xMotion = joint.yMotion = joint.zMotion = ConfigurableJointMotion.Limited;
//             joint.angularXMotion = joint.angularYMotion = joint.angularZMotion = ConfigurableJointMotion.Limited;
//             joint.autoConfigureConnectedAnchor = false;
//             joint.connectedAnchor = Vector3.zero;

//             JointDrive drive = new JointDrive { positionSpring = 0, positionDamper = 0, maximumForce = Mathf.Infinity };
//             joint.xDrive = joint.yDrive = joint.zDrive = drive;

//             float targetSpring = 100; // Target stiffness
//             float targetDamper = 5; // Target damping
//             float attachingObjectDuration = 5f;
//             float elapsedTimeForAttachingObject = 0f;

//             while (elapsedTimeForAttachingObject < attachingObjectDuration)
//             {
//                 elapsedTimeForAttachingObject += Time.deltaTime;
//                 float t = elapsedTimeForAttachingObject / attachingObjectDuration;

//                 float spring = Mathf.Lerp(0, targetSpring, t);
//                 float damper = Mathf.Lerp(0, targetDamper, t);

//                 drive.positionSpring = spring;
//                 drive.positionDamper = damper;
//                 joint.xDrive = joint.yDrive = joint.zDrive = drive;

//                 yield return null;
//             }
//         }
//     }

//     private void Update()
//     {
//         UpdateCranePositionFlags(); // Update flags at the start of each frame
        
//         if (canGrab && !objectReleased)
//         {
//             Ray ray = new Ray(transform.position, -transform.right);
//             RaycastHit hit;

//             if (Physics.Raycast(ray, out hit, grabReleaseRange, grabbableLayer) && isGrabbingButtonHeld)
//             {
//                 buttonHoldTime += Time.deltaTime;
//                 // Check if the button has been held for more than 2 seconds
//                 if (buttonHoldTime >= 5f)
//                 {
//                     grabbedObject = hit.collider.transform;
//                     StartCoroutine(AttachAfterDelay(grabbedObject));
//                     canGrab = false; // Prevents grabbing in the same frame
//                     // Reset hold time to avoid multiple triggers
//                     buttonHoldTime = 0;
//                     isGrabbingButtonHeld = false;
//                 }
//             }
//         }

//         if (grabbedObject != null && !objectReleased)
//         {
//             // Check if the grabbed object is above the unloading area after attaching
//             RaycastHit hitFromAttached;

//             if (Physics.Raycast(grabbedObject.position, Vector3.down, out hitFromAttached, grabReleaseRange, unloadingStoreyLayer) && isReleasingButtonHeld)
//             {
//                 buttonHoldTime += Time.deltaTime;
//                 // Check if the button has been held for more than 2 seconds
//                 if (buttonHoldTime >= 5f)
//                 {
//                     // Release the object from the crane hook by removing the configurable joint
//                     Destroy(grabbedObject.GetComponent<ConfigurableJoint>());
//                     grabbedObject.SetParent(null);
//                     grabbedObject = null;
//                     objectReleased = true;
//                     // canGrab = true; // Allow grabbing in the next frame
//                     unloadingFinished = true;

//                     // Reset hold time to avoid multiple triggers
//                     buttonHoldTime = 0;
//                     isGrabbingButtonHeld = false;
//                 }
//             }
//         }
//     }

//     private void UpdateCranePositionFlags()
//     {
//         // Update craneHookAboveGrabbableObject and craneHookAboveUnloadingArea here based on your raycasting logic
//         Ray ray = new Ray(transform.position, -transform.right);
//         RaycastHit hit;

//         craneHookAboveGrabbableObject = canGrab && grabbedObject == null && Physics.Raycast(ray, out hit, grabReleaseRange, grabbableLayer);

//         Ray rayDown = new Ray(grabbedObject?.position ?? Vector3.zero, Vector3.down);
//         craneHookAboveUnloadingStorey = grabbedObject != null && Physics.Raycast(rayDown, out hit, grabReleaseRange, unloadingStoreyLayer);
//     }

//     private void StartGrabbingButtonHold()
//     {
//         isGrabbingButtonHeld = true;
//         buttonHoldTime = 0; // Reset the timer when button press starts
//         if (craneHookAboveGrabbableObject){
//             progressBarManager.StartButtonHold(); // Start the progress bar
//             progressBarManagerBlind.StartButtonHold();
//             chainAudioSource.PlayOneShot(chainSoundEffect);
//         }
//     }

//     private void EndGrabbingButtonHold()
//     {
//         isGrabbingButtonHeld = false;
//         buttonHoldTime = 0; // Reset the timer when button is released
//         progressBarManager.EndButtonHold(); // Hide the progress bar
//         progressBarManagerBlind.EndButtonHold(); // Hide the progress bar
//         chainAudioSource.Stop(); // Stop the chain sound when grabbing is cancelled or completed
//     }

//     private void StartReleasingButtonHold()
//     {
//         isReleasingButtonHeld = true;
//         buttonHoldTime = 0; // Reset the timer when button press starts
//         if (craneHookAboveUnloadingStorey){
//             progressBarManager.StartButtonHold(); // Start the progress bar
//             progressBarManagerBlind.StartButtonHold(); // Start the progress bar
//             chainAudioSource.PlayOneShot(chainSoundEffect);
//         }
//     }

//     private void EndReleasingButtonHold()
//     {
//         isReleasingButtonHeld = false;
//         buttonHoldTime = 0; // Reset the timer when button is released
//         progressBarManager.EndButtonHold(); // Hide the progress bar
//         progressBarManagerBlind.EndButtonHold(); // Hide the progress bar
//         chainAudioSource.Stop();
//     }

//     void OnDestroy()
//     {
//         // Unsubscribe to prevent memory leaks
//         objectMovement.GrabOrReleaseObject.GrabObject.performed -= ctx => StartGrabbingButtonHold();
//         objectMovement.GrabOrReleaseObject.GrabObject.canceled -= ctx => EndGrabbingButtonHold();
//         objectMovement.GrabOrReleaseObject.GrabObject.performed -= ctx => StartReleasingButtonHold();
//         objectMovement.GrabOrReleaseObject.GrabObject.canceled -= ctx => EndReleasingButtonHold();
//         objectMovement.Disable(); // Disable the input action map
//     }
// }


//////////////////////////////////////////////////////////////////////

// using System.Collections;
// using UnityEngine;

// public class MoveObject : MonoBehaviour
// {
//     private LayerMask grabbableLayer;
//     private LayerMask unloadingAreaLayer;
//     private LayerMask chainLinkedBelowTheRopeLayer;

//     private float grabReleaseRange = 15f;
//     private Transform grabbedObject;
//     private float loadingTime = 5f;
//     private bool objectReleased = false;
//     private float aboveUnloadingAreaTime = 0f;
//     private float releaseDelay = 5f;

//     public bool canGrab = true;
//     public bool unloadingFinished = false;

//     void Start()
//     {
//         grabbableLayer = 1 << LayerMask.NameToLayer("GrabbableObject");
//         unloadingAreaLayer = 1 << LayerMask.NameToLayer("UnloadingArea");
//         chainLinkedBelowTheRopeLayer = 1 << LayerMask.NameToLayer("linkBelowTheRope");
//     }

//     IEnumerator AttachAfterDelay(Transform objectToAttach)
//     {
//         yield return new WaitForSeconds(loadingTime);
//         objectToAttach.SetParent(transform);

//         Collider[] colliders = Physics.OverlapSphere(transform.position, 500f, chainLinkedBelowTheRopeLayer);
//         if (colliders.Length > 0)
//         {
//             Transform lowestLink = colliders[0].transform;
//             objectToAttach.position = lowestLink.position; // Initially position object correctly

//             // Create a joint and initially set it to be very 'soft'
//             ConfigurableJoint joint = objectToAttach.gameObject.AddComponent<ConfigurableJoint>();
//             joint.connectedBody = lowestLink.GetComponent<Rigidbody>();
//             joint.xMotion = joint.yMotion = joint.zMotion = ConfigurableJointMotion.Limited;
//             joint.angularXMotion = joint.angularYMotion = joint.angularZMotion = ConfigurableJointMotion.Limited;
//             joint.autoConfigureConnectedAnchor = false;
//             joint.connectedAnchor = Vector3.zero;

//             JointDrive drive = new JointDrive { positionSpring = 0, positionDamper = 0, maximumForce = Mathf.Infinity };
//             joint.xDrive = joint.yDrive = joint.zDrive = drive;

//             float targetSpring = 100; // Target stiffness
//             float targetDamper = 5; // Target damping
//             float attachingObjectDuration = 5f;
//             float elapsedTimeForAttachingObject = 0f;

//             while (elapsedTimeForAttachingObject < attachingObjectDuration)
//             {
//                 elapsedTimeForAttachingObject += Time.deltaTime;
//                 float t = elapsedTimeForAttachingObject / attachingObjectDuration;

//                 float spring = Mathf.Lerp(0, targetSpring, t);
//                 float damper = Mathf.Lerp(0, targetDamper, t);

//                 drive.positionSpring = spring;
//                 drive.positionDamper = damper;
//                 joint.xDrive = joint.yDrive = joint.zDrive = drive;

//                 yield return null;
//             }
//         }
//     }

//     private void Update()
//     {
//         if (canGrab && !objectReleased)
//         {
//             Ray ray = new Ray(transform.position, -transform.right);
//             RaycastHit hit;

//             if (Physics.Raycast(ray, out hit, grabReleaseRange, grabbableLayer))
//             {
//                 grabbedObject = hit.collider.transform;
//                 StartCoroutine(AttachAfterDelay(grabbedObject));
//                 canGrab = false; // Prevents grabbing in the same frame
//             }
//         }

//         if (grabbedObject != null && !objectReleased)
//         {
//             // Check if the grabbed object is above the unloading area after attaching
//             RaycastHit hitFromAttached;

//             if (Physics.Raycast(grabbedObject.position, Vector3.down, out hitFromAttached, grabReleaseRange, unloadingAreaLayer))
//             {
//                 // Increment timer if above unloading area
//                 aboveUnloadingAreaTime += Time.deltaTime;

//                 if (aboveUnloadingAreaTime >= releaseDelay)
//                 {
//                     // Release the object from the crane hook by removing the configurable joint
//                     Destroy(grabbedObject.GetComponent<ConfigurableJoint>());
//                     grabbedObject.SetParent(null);
//                     grabbedObject = null;
//                     objectReleased = true;
//                     canGrab = true; // Allow grabbing in the next frame
//                     aboveUnloadingAreaTime = 0f; // Reset timer
//                     unloadingFinished = true;
//                 }
//             }
//             else
//             {
//                 aboveUnloadingAreaTime = 0f; // Reset timer if not above unloading area
//             }
//         }
//     }
// }




// Scenario: Moving the object with two chains
// using System.Collections;
// using UnityEngine;

// public class MoveObject : MonoBehaviour
// {
//     private LayerMask grabbableLayer;
//     private LayerMask unloadingAreaLayer;
//     private LayerMask chainLinkedBelowTheRopeLayer1;
//     private LayerMask chainLinkedBelowTheRopeLayer2;

//     private float grabReleaseRange = 15f;
//     private Transform grabbedObject;
//     private float loadingTime = 5f;
//     private bool objectReleased = false;
//     private float aboveUnloadingAreaTime = 0f;
//     private float releaseDelay = 5f;

//     public bool canGrab = true;
//     public bool unloadingFinished = false;

//     void Start()
//     {
//         grabbableLayer = 1 << LayerMask.NameToLayer("GrabbableObject");
//         unloadingAreaLayer = 1 << LayerMask.NameToLayer("UnloadingArea");
//         chainLinkedBelowTheRopeLayer1 = 1 << LayerMask.NameToLayer("linkBelowTheRope1");
//         chainLinkedBelowTheRopeLayer2 = 1 << LayerMask.NameToLayer("linkBelowTheRope2");
//     }

//     IEnumerator AttachAfterDelay(Transform objectToAttach)
//     {
//         yield return new WaitForSeconds(loadingTime);

//         // Find attachment points as children of the grabbable object
//         Transform attachmentPoint1 = objectToAttach.Find("AttachmentPoint1");
//         Transform attachmentPoint2 = objectToAttach.Find("AttachmentPoint2");

//         // Attach to chain 1 at attachment point 1
//         Collider[] collidersChain1 = Physics.OverlapSphere(attachmentPoint1.position, 500f, chainLinkedBelowTheRopeLayer1);
//         if (collidersChain1.Length > 0)
//         {
//             CreateConfigurableJoint(objectToAttach, attachmentPoint1.localPosition, collidersChain1[0].transform);
//         }

//         // Attach to chain 2 at attachment point 2
//         Collider[] collidersChain2 = Physics.OverlapSphere(attachmentPoint2.position, 500f, chainLinkedBelowTheRopeLayer2);
//         if (collidersChain2.Length > 0)
//         {
//             CreateConfigurableJoint(objectToAttach, attachmentPoint2.localPosition, collidersChain2[0].transform);
//         }
//     }

//     ConfigurableJoint CreateConfigurableJoint(Transform grabbableObject, Vector3 localAnchor, Transform link)
//     {
//         ConfigurableJoint joint = grabbableObject.gameObject.AddComponent<ConfigurableJoint>();
//         joint.connectedBody = link.GetComponent<Rigidbody>();
//         joint.anchor = localAnchor;
//         joint.autoConfigureConnectedAnchor = false;
//         joint.connectedAnchor = Vector3.zero;

//         joint.xMotion = joint.yMotion = joint.zMotion = ConfigurableJointMotion.Limited;
//         joint.angularXMotion = joint.angularYMotion = joint.angularZMotion = ConfigurableJointMotion.Limited;

//         return joint;
//     }

//     private void Update()
//     {
//         if (canGrab && !objectReleased)
//         {
//             Ray ray = new Ray(transform.position, -transform.right);
//             RaycastHit hit;
//             if (Physics.Raycast(ray, out hit, grabReleaseRange, grabbableLayer))
//             {
//                 grabbedObject = hit.collider.transform;
//                 StartCoroutine(AttachAfterDelay(grabbedObject));
//                 canGrab = false; // Prevent grabbing in the same frame
//             }
//         }

//         if (grabbedObject != null && !objectReleased)
//         {
//             RaycastHit hitFromAttached;
//             if (Physics.Raycast(grabbedObject.position, Vector3.down, out hitFromAttached, grabReleaseRange, unloadingAreaLayer))
//             {
//                 aboveUnloadingAreaTime += Time.deltaTime;
//                 if (aboveUnloadingAreaTime >= releaseDelay)
//                 {
//                     // Detach from all joints
//                     foreach (var joint in grabbedObject.GetComponents<ConfigurableJoint>())
//                     {
//                         Destroy(joint);
//                     }
//                     grabbedObject.SetParent(null);
//                     grabbedObject = null;
//                     objectReleased = true;
//                     canGrab = true;
//                     aboveUnloadingAreaTime = 0f;
//                     unloadingFinished = true;
//                 }
//             }
//             else
//             {
//                 aboveUnloadingAreaTime = 0f; // Reset timer if not above unloading area
//             }
//         }
//     }
// }


////////////////////////////////////////////////////////////////////////////
/// The grabbable object suudenly attaches to the crane hook in the following version of the code

// using System.Collections;
// using UnityEngine;

// public class MoveObject : MonoBehaviour
// {
//     private LayerMask grabbableLayer;
//     private LayerMask unloadingAreaLayer;
//     private LayerMask chainLinkedBelowTheRopeLayer;

//     private float grabReleaseRange = 15f;
//     private Transform grabbedObject;
//     private float loadingTime = 5f;
//     private bool objectReleased = false;
//     private float aboveUnloadingAreaTime = 0f;
//     private float releaseDelay = 5f;

//     public bool canGrab = true;
//     public bool unloadingFinished = false;

//     void Start()
//     {
//         grabbableLayer = 1 << LayerMask.NameToLayer("GrabbableObject");
//         unloadingAreaLayer = 1 << LayerMask.NameToLayer("UnloadingArea");
//         chainLinkedBelowTheRopeLayer = 1 << LayerMask.NameToLayer("linkBelowTheRope");
//     }

//     IEnumerator AttachAfterDelay(Transform objectToAttach)
//     {
//         yield return new WaitForSeconds(loadingTime);

//         // Find the lowest link with the "chainLinkedBelowTheRope" layer
//         Collider[] colliders = Physics.OverlapSphere(transform.position, 500f, chainLinkedBelowTheRopeLayer);

//         if (colliders.Length > 0)
//         {
//             Transform lowestLink = colliders[0].transform;

//             // Attach the grabbed object to the lowest link using a configurable joint
//             ConfigurableJoint joint = objectToAttach.gameObject.AddComponent<ConfigurableJoint>();
//             joint.connectedBody = lowestLink.GetComponent<Rigidbody>();
//             joint.autoConfigureConnectedAnchor = false;
//             joint.connectedAnchor = Vector3.zero;

//             // Set motion constraints to "Limited"
//             SoftJointLimit limit = new SoftJointLimit { limit = 0.1f };
//             joint.linearLimit = limit;
//             joint.angularXMotion = ConfigurableJointMotion.Limited;
//             joint.angularYMotion = ConfigurableJointMotion.Limited;
//             joint.angularZMotion = ConfigurableJointMotion.Limited;
//             joint.xMotion = ConfigurableJointMotion.Limited;
//             joint.yMotion = ConfigurableJointMotion.Limited;
//             joint.zMotion = ConfigurableJointMotion.Limited;
//         }
//     }

//     private void Update()
//     {
//         if (canGrab && !objectReleased)
//         {
//             Ray ray = new Ray(transform.position, -transform.right);
//             RaycastHit hit;

//             if (Physics.Raycast(ray, out hit, grabReleaseRange, grabbableLayer))
//             {
//                 grabbedObject = hit.collider.transform;
//                 StartCoroutine(AttachAfterDelay(grabbedObject));
//                 canGrab = false; // Prevents grabbing in the same frame
//             }
//         }

//         if (grabbedObject != null && !objectReleased)
//         {
//             // Check if the grabbed object is above the unloading area after attaching
//             RaycastHit hitFromAttached;

//             if (Physics.Raycast(grabbedObject.position, Vector3.down, out hitFromAttached, grabReleaseRange, unloadingAreaLayer))
//             {
//                 // Increment timer if above unloading area
//                 aboveUnloadingAreaTime += Time.deltaTime;

//                 if (aboveUnloadingAreaTime >= releaseDelay)
//                 {
//                     // Release the object from the crane hook by removing the configurable joint
//                     Destroy(grabbedObject.GetComponent<ConfigurableJoint>());
//                     grabbedObject = null;
//                     objectReleased = true;
//                     canGrab = true; // Allow grabbing in the next frame
//                     aboveUnloadingAreaTime = 0f; // Reset timer
//                     unloadingFinished = true;
//                 }
//             }
//             else
//             {
//                 aboveUnloadingAreaTime = 0f; // Reset timer if not above unloading area
//             }
//         }
//     }
// }





// using System.Collections;
// using UnityEngine;

// public class MoveObject : MonoBehaviour
// {
//     private LayerMask grabbableLayer;
//     private LayerMask unloadingAreaLayer;

//     private float grabReleaseRange = 5f; // If you changed this variable, also change it in ground_person script
//     private Transform grabbedObject;
//     private float loadingTime = 5f;
//     private bool objectReleased = false;
//     private float aboveUnloadingAreaTime = 0f;
//     private float releaseDelay = 5f;

//     public bool canGrab = true;
//     public bool unloadingFinished = false;

//     void Start()
//     {
//         grabbableLayer = 1 << LayerMask.NameToLayer("GrabbableObject");
//         unloadingAreaLayer = 1 << LayerMask.NameToLayer("UnloadingArea");
//     }

//     IEnumerator AttachAfterDelay(Transform objectToAttach)
//     {
//         yield return new WaitForSeconds(loadingTime);

//         // Attach the grabbed object to the crane hook
//         ConfigurableJoint joint = objectToAttach.gameObject.AddComponent<ConfigurableJoint>();
//         joint.connectedBody = GetComponent<Rigidbody>();
//         joint.autoConfigureConnectedAnchor = false;
//         joint.connectedAnchor = Vector3.zero;

//         // Disable gravity or any physics on the grabbed object (if needed)
//         Rigidbody grabbedRB = objectToAttach.GetComponent<Rigidbody>();
//         if (grabbedRB != null)
//         {
//             grabbedRB.useGravity = false;
//         }
//     }

//     private void Update()
//     {
//         if (canGrab && !objectReleased)
//         {
//             Ray ray = new Ray(transform.position, -transform.right);
//             RaycastHit hit;

//             if (Physics.Raycast(ray, out hit, grabReleaseRange, grabbableLayer))
//             {
//                 grabbedObject = hit.collider.transform;
//                 StartCoroutine(AttachAfterDelay(grabbedObject));
//                 canGrab = false; // Prevents grabbing in the same frame
//             }
//         }

//         if (grabbedObject != null && !objectReleased)
//         {
//             // Check if the grabbed object is above the unloading area after attaching
//             RaycastHit hitFromAttached;

//             if (Physics.Raycast(grabbedObject.position, Vector3.down, out hitFromAttached, grabReleaseRange, unloadingAreaLayer))
//             {
//                 // Increment timer if above unloading area
//                 aboveUnloadingAreaTime += Time.deltaTime;

//                 if (aboveUnloadingAreaTime >= releaseDelay)
//                 {
//                     // Restore any physics properties if needed (if you disabled gravity)
//                     Rigidbody grabbedRB = grabbedObject.GetComponent<Rigidbody>();
//                     if (grabbedRB != null)
//                     {
//                         grabbedRB.useGravity = true;
//                     }

//                     // Release the object from the crane hook by breaking the joint
//                     Destroy(grabbedObject.GetComponent<ConfigurableJoint>());
//                     grabbedObject.SetParent(null);
//                     grabbedObject = null;
//                     objectReleased = true;
//                     canGrab = true; // Allow grabbing in the next frame
//                     aboveUnloadingAreaTime = 0f; // Reset timer
//                     unloadingFinished = true;
//                 }
//             }
//             else
//             {
//                 aboveUnloadingAreaTime = 0f; // Reset timer if not above unloading area
//             }
//         }
//     }
// }





// using System.Collections;
// using UnityEngine;

// public class MoveObject : MonoBehaviour
// {
//     private LayerMask grabbableLayer;
//     private LayerMask unloadingAreaLayer;

//     private float grabReleaseRange = 5f; // If you changed this variable, also change it in ground_person script
//     private Transform grabbedObject;
//     private float loadingTime = 5f;
//     private bool objectReleased = false;
//     private float aboveUnloadingAreaTime = 0f;
//     private float releaseDelay = 5f;

//     public bool canGrab = true;
//     public bool unloadingFinished = false;

//     void Start()
//     {
//         grabbableLayer = 1 << LayerMask.NameToLayer("GrabbableObject");
//         unloadingAreaLayer = 1 << LayerMask.NameToLayer("UnloadingArea");
//     }

//     IEnumerator AttachAfterDelay(Transform objectToAttach)
//     {
//         yield return new WaitForSeconds(loadingTime);

//         // Attach the grabbed object to the crane hook
//         objectToAttach.SetParent(transform);

//         // Disable gravity or any physics on the grabbed object
//         Rigidbody grabbedRB = objectToAttach.GetComponent<Rigidbody>();
//         if (grabbedRB != null)
//         {
//             grabbedRB.isKinematic = true;
//         }
//     }

//     private void Update()
//     {
//         if (canGrab && !objectReleased)
//         {
//             Ray ray = new Ray(transform.position, -transform.right);
//             RaycastHit hit;

//             if (Physics.Raycast(ray, out hit, grabReleaseRange, grabbableLayer))
//             {
//                 grabbedObject = hit.collider.transform;
//                 StartCoroutine(AttachAfterDelay(grabbedObject));
//                 canGrab = false; // Prevents grabbing in the same frame
//             }
//         }

//         if (grabbedObject != null && !objectReleased)
//         {
//             // Check if the grabbed object is above the unloading area after attaching
//             RaycastHit hitFromAttached;

//             if (Physics.Raycast(grabbedObject.position, Vector3.down, out hitFromAttached, grabReleaseRange, unloadingAreaLayer))
//             {
//                 // Increment timer if above unloading area
//                 aboveUnloadingAreaTime += Time.deltaTime;

//                 if (aboveUnloadingAreaTime >= releaseDelay)
//                 {
//                     // Restore any physics properties if needed
//                     Rigidbody grabbedRB = grabbedObject.GetComponent<Rigidbody>();
//                     if (grabbedRB != null)
//                     {
//                         grabbedRB.isKinematic = false;
//                         // Release the object from the crane hook
//                         grabbedObject.SetParent(null);
//                         grabbedObject = null;
//                         objectReleased = true;
//                         canGrab = true; // Allow grabbing in the next frame
//                         aboveUnloadingAreaTime = 0f; // Reset timer
//                         unloadingFinished = true;
//                     }
//                 }
//             }
//             else
//             {
//                 aboveUnloadingAreaTime = 0f; // Reset timer if not above unloading area
//             }
//         }
//     }
// }









// using System.Collections;
// using UnityEngine;

// public class MoveObject : MonoBehaviour
// {
//     private LayerMask grabbableLayer;
//     private LayerMask unloadingAreaLayer;

//     private float grabReleaseRange = 2f;
//     private Transform grabbedObject;
//     private float loadingTime = 5f;
//     private bool objectReleased = false;
//     private float aboveUnloadingAreaTime = 0f;
//     private float releaseDelay = 5f;

//     public bool canGrab = true;
//     public bool unloadingFinished = false;

//     private HingeJoint joint;

//     void Start()
//     {
//         grabbableLayer = 1 << LayerMask.NameToLayer("GrabbableObject");
//         unloadingAreaLayer = 1 << LayerMask.NameToLayer("UnloadingArea");
//     }

//     IEnumerator AttachAfterDelay(Transform objectToAttach)
//     {
//         yield return new WaitForSeconds(loadingTime);

//         // Attach the grabbed object to the crane hook
//         // objectToAttach.SetParent(transform);

//         // Add ConfigurableJoint to simulate swing
//         joint = objectToAttach.gameObject.AddComponent<HingeJoint>();
//         joint.connectedBody = transform.GetComponent<Rigidbody>();
//         // joint.xMotion = ConfigurableJointMotion.Limited;
//         // joint.yMotion = ConfigurableJointMotion.Locked;
//         // joint.zMotion = ConfigurableJointMotion.Limited;
//         // joint.angularXMotion = ConfigurableJointMotion.Limited;
//         // joint.angularYMotion = ConfigurableJointMotion.Locked;
//         // joint.angularZMotion = ConfigurableJointMotion.Limited; // Allow rotation around Z-axis
//     }

//     private void Update()
//     {
//         if (canGrab && !objectReleased)
//         {
//             Ray ray = new Ray(transform.position, -transform.right);
//             RaycastHit hit;

//             if (Physics.Raycast(ray, out hit, grabReleaseRange, grabbableLayer))
//             {
//                 Debug.DrawRay(transform.position, -transform.right * hit.distance, Color.red);
//                 grabbedObject = hit.collider.transform;
//                 StartCoroutine(AttachAfterDelay(grabbedObject));
//                 canGrab = false; // Prevents grabbing in the same frame
//             }
//             else
//             {
//                 Debug.DrawRay(transform.position, -transform.right * 100f, Color.green);
//             }
//         }

//         if (grabbedObject != null && !objectReleased)
//         {
//             // Check if the grabbed object is above the unloading area after attaching
//             RaycastHit hitFromAttached;

//             if (Physics.Raycast(grabbedObject.position, Vector3.down, out hitFromAttached, Mathf.Infinity, unloadingAreaLayer))
//             {
//                 Debug.DrawRay(grabbedObject.position, Vector3.down * hitFromAttached.distance, Color.yellow);

//                 // Increment timer if above unloading area
//                 aboveUnloadingAreaTime += Time.deltaTime;

//                 if (aboveUnloadingAreaTime >= releaseDelay)
//                 {
//                     // Remove ConfigurableJoint when releasing the object
//                     Destroy(joint);

//                     // Release the object from the crane hook
//                     grabbedObject.SetParent(null);
//                     grabbedObject = null;
//                     objectReleased = true;
//                     canGrab = true; // Allow grabbing in the next frame
//                     aboveUnloadingAreaTime = 0f; // Reset timer
//                     unloadingFinished = true;
//                 }
//             }
//             else
//             {
//                 Debug.DrawRay(grabbedObject.position, Vector3.down * 100f, Color.blue);
//                 aboveUnloadingAreaTime = 0f; // Reset timer if not above unloading area
//             }
//         }
//     }
// }
