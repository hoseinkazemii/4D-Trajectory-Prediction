using UnityEngine;
using System.Collections.Generic;
using System.IO;

public class crane_animate2 : MonoBehaviour
{
    CraneActions craneActions;

    public Animator animator;
    public float dolly;
    public float hook;
    private float rotateYaw;
    public float adjustedRotateYaw = 180f; // Make it public
    private float movementSpeed = 3f; // Adjust movementSpeed to make it feel more realistic
    private float minimumValueForTrolleyHookMovement = 0.2f;
    private float currentRotateSpeed = 0f;
    private float acceleration = 2f; // Adjust acceleration for smooth start and stop

    public CabinPanelManager cabinPanelManager;
    public WindArea windArea;

    private Vector2 trolleyAndHookMovement;
    private Vector2 rotate;

    private bool taskStarted = false;
    public static float taskStartTime;

    void Awake()
    {
        dolly = 10f;
        craneActions = new CraneActions();
        craneActions.CraneMovement.RotateCrane.performed += ctx => rotate = ctx.ReadValue<Vector2>();
        craneActions.CraneMovement.RotateCrane.canceled += ctx => rotate = Vector2.zero;
        craneActions.CraneMovement.MoveTrolley.performed += ctx => trolleyAndHookMovement = ctx.ReadValue<Vector2>();
        craneActions.CraneMovement.MoveTrolley.canceled += ctx => trolleyAndHookMovement.y = 0f;
        craneActions.CraneMovement.MoveTrolley.canceled += ctx => trolleyAndHookMovement.x = 0f;
    }

    void Update()
    {
        if (!taskStarted && (rotate != Vector2.zero || trolleyAndHookMovement != Vector2.zero))
        {
            taskStartTime = Time.time;
            taskStarted = true;
        }

        rotateYaw = (rotateYaw + 360) % 360;
        animator.SetFloat("Rotate_YAW", rotateYaw);
        adjustedRotateYaw = (rotateYaw - 180 + 360) % 360;

        float targetRotateSpeed = rotate.x * movementSpeed;
        currentRotateSpeed = Mathf.MoveTowards(currentRotateSpeed, targetRotateSpeed, acceleration * 0.005f);
        rotateYaw -= currentRotateSpeed * Time.deltaTime;

        if (Mathf.Abs(trolleyAndHookMovement.x) >= minimumValueForTrolleyHookMovement)
        {
            dolly = Mathf.Clamp(dolly + trolleyAndHookMovement.x * Time.deltaTime * movementSpeed, 5f, 100f);
        }

        if (Mathf.Abs(trolleyAndHookMovement.y) >= minimumValueForTrolleyHookMovement)
        {
            hook = Mathf.Clamp(hook + trolleyAndHookMovement.y * Time.deltaTime * movementSpeed, 0f, 57f);
        }

        animator.SetFloat("dolly", dolly);
        animator.SetFloat("hook", hook);

        float trolleyPosition = dolly;
        cabinPanelManager.SetTrolleyPosition(trolleyPosition);

        float hookPosition = hook;
        cabinPanelManager.SetHook(hookPosition);

        float Radius = adjustedRotateYaw;
        cabinPanelManager.SetRadius(Radius);

        float windSpeed = windArea.windStrength * 5f;
        cabinPanelManager.SetWindSpeed(windSpeed);
    }

    void OnEnable()
    {
        craneActions.CraneMovement.Enable();
    }

    void OnDisable()
    {
        craneActions.CraneMovement.Disable();
    }
}
